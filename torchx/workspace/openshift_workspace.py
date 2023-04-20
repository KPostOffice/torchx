# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import posixpath
import stat
import sys
import tarfile
import tempfile
from typing import Dict, IO, Iterable, Mapping, Optional, TextIO, Tuple, TYPE_CHECKING
import hashlib
import yaml
import subprocess

import fsspec

import torchx
from torchx.specs import AppDef, CfgVal, Role, runopts
from torchx.workspace.api import walk_workspace, WorkspaceMixin
from torchx.workspace import openshift_templates

import openshift as oc

# TO DELETE: allow push location to be internal image registry (ImageStream) or external (dockerhub, quay, etc.)
# TO DELETE: the external case assumes that there is the requisite push secret in the namespace.

# TODO: create build config that pushes to tag which is the archives md5 hashfffff


if TYPE_CHECKING:
    from docker import DockerClient

log: logging.Logger = logging.getLogger(__name__)


TORCHX_DOCKERFILE = "Dockerfile"

DEFAULT_DOCKERFILE = b"""
ARG IMAGE
FROM $IMAGE

COPY . .
"""


class OpenShiftWorkspaceMixin(WorkspaceMixin[Dict[str, Tuple[str, str]]]):
    """
    DockerWorkspaceMixin will build patched docker images from the workspace. These
    patched images are docker images and can be either used locally via the
    docker daemon or pushed using the helper methods to a remote repository for
    remote jobs.

    This requires a running docker daemon locally and for remote pushing
    requires being authenticated to those repositories via ``docker login``.

    If there is a ``Dockerfile.torchx`` file present in the workspace that will
    be used instead to build the container.

    The docker build is provided with some extra build arguments that can be
    used in the Dockerfile.torchx:

    * IMAGE: the image string from the first Role in the AppDef
    * WORKSPACE: the full workspace path

    To exclude files from the build context you can use the standard
    `.dockerignore` file.

    See more:

    * https://docs.docker.com/engine/reference/commandline/login/
    * https://docs.docker.com/get-docker/
    """

    LABEL_VERSION: str = "torchx.pytorch.org/version"

    def __init__(
        self,
        *args: object,
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)

    def workspace_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "image_repo",
            type_=str,
            help="(remote jobs) the image repository to use when pushing patched images, must have push access. Ex: example.com/your/container",
        )
        return opts

    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """
        Builds a new docker image using the ``role``'s image as the base image
        and updates the ``role``'s image with this newly built docker image id

        Args:
            role: the role whose image (a Docker image) is to be used as the base image
            workspace: a fsspec path to a directory with contents to be overlaid
        """

        context = _build_context(role.image, workspace)
        role._base_image = role.image
        role.image = context.name

    def dryrun_push_images(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> Dict[str, Tuple[str, str, str]]:
        """
        _update_app_images replaces the local Docker images (identified via
        ``sha256:...``) in the provided ``AppDef`` with the remote path that they will be uploaded to and
        returns a mapping of local to remote names.

        ``push`` must be called with the returned mapping before
        launching the job.

        Returns:
            A dict of [local image name, (remote repo, tag)].
        """
        image_repo = cfg.get("image_repo")

        images_to_push = {}
        for role in app.roles:
            if isinstance(role.image, str):
                if not image_repo:
                    raise KeyError(
                        f"must specify the image repository via `image_repo` config to be able to upload local image {role.image}"
                    )
                assert isinstance(image_repo, str), "image_repo must be str"

                # generate build config

                with open(role.image, "rb") as archive:
                    archive_hash = _get_md5_checksum(archive)
                output = openshift_templates.IMAGE_STREAM_OUTPUT.format(
                    image_repo=image_repo, sha=archive_hash
                )
                if cfg.get("image_secret"):
                    image_secret = openshift_templates.PULL_SECRET.format(
                        image_secret=cfg["image_secret"]
                    )
                else:
                    image_secret = ""

                namespace = cfg.get("namespace", oc.get_project_name())

                build_args = openshift_templates.BUILD_ARGS.format(image=role._base_image)  # TODO check what role.base_image contains

                build_config = openshift_templates.BUILD_CONFIG_TEMPLATE.format(
                    image_repo=image_repo,
                    sha=archive_hash,
                    namespace=namespace,
                    version=torchx.version.__version__,
                    output=output,
                    image_secret=image_secret,
                    build_args=build_args,  # TODO construct build args
                )
                remote_image = f"image-registry.openshift-image-registry.svc:5000/{namespace}/{image_repo}:{archive_hash}"
                images_to_push[remote_image] = (
                    image_repo,
                    archive_hash,
                    build_config,
                    role.image,
                )
                role.image = remote_image
        return images_to_push

    def push_images(self, images_to_push: Dict[IO[bytes], Tuple[str, str]]) -> None:
        """
        _push_images pushes the specified images to the remote container
        repository with the specified tag. The docker daemon must be
        authenticated to the remote repository using ``docker login``.

        Args:
            images_to_push: A dict of [local image name, (remote repo, tag)].
        """

        if len(images_to_push) == 0:
            return

        for _, (repo, tag, build_config, file_name) in images_to_push.items():
            with open(file_name, "rb") as file:
                log.info(f"pushing image {repo}:{tag}...")
                build_config_dict = yaml.safe_load(build_config)
                namespace = build_config_dict["metadata"]["namespace"]
                oc.apply(build_config_dict)
                subprocess.run(f"oc start-build -n {namespace} {build_config_dict['metadata']['name']} --from-archive={file_name} --follow", shell=True)
                # oc.start_build(
                #     [
                #         f"-n {namespace}",
                #         f"{build_config_dict['metadata']['name']}",
                #         f"--from-archive={file.name}"],
                # )


def _build_context(img: str, workspace: str) -> IO[bytes]:
    # f is closed by parent, NamedTemporaryFile auto closes on GC
    f = tempfile.NamedTemporaryFile(  # noqa P201
        prefix="torchx-context",
        suffix=".tar",
        delete=False,
    )

    with tarfile.open(fileobj=f, mode="w") as tf:
        _copy_to_tarfile(workspace, tf)
        if TORCHX_DOCKERFILE not in tf.getnames():
            info = tarfile.TarInfo(TORCHX_DOCKERFILE)
            info.size = len(DEFAULT_DOCKERFILE)
            tf.addfile(info, io.BytesIO(DEFAULT_DOCKERFILE))
    f.seek(0)
    return f


def _copy_to_tarfile(workspace: str, tf: tarfile.TarFile) -> None:
    fs, path = fsspec.core.url_to_fs(workspace)
    log.info(f"Workspace `{workspace}` resolved to filesystem path `{path}`")
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in walk_workspace(fs, path, ".dockerignore"):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            with fs.open(info["name"], "rb") as f:
                filepath = posixpath.join(relpath, file) if relpath != "." else file
                tinfo = tarfile.TarInfo(filepath)
                size = info["size"]
                assert isinstance(size, int), "size must be an int"
                tinfo.size = size

                # preserve unix mode for supported filesystems; fsspec.filesystem("memory") for example does not support
                # unix file mode, hence conditional check here
                if "mode" in info:
                    mode = info["mode"]
                    assert isinstance(mode, int), "mode must be an int"
                    tinfo.mode = stat.S_IMODE(mode)

                tf.addfile(tinfo, f)


def _get_md5_checksum(f: IO):
    hash_md5 = hashlib.md5()
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.seek(0)
    return hash_md5.hexdigest()
