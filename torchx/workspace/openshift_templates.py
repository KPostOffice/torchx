# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

DOCKER_OUTPUT = """
      kind: 'DockerImage'
      name: '{image_repo}:{sha}'
"""

IMAGE_STREAM_OUTPUT = """
      kind: 'ImageStreamTag'
      name: '{image_repo}:{sha}'
"""

PULL_SECRET = """
    pullSecret:
      name: {pull_secret_name}
"""

BUILD_ARGS = """
        - name: IMAGE
          value: {image}
"""

BUILD_CONFIG_TEMPLATE = """
kind: BuildConfig
apiVersion: build.openshift.io/v1
metadata:
  name: {image_repo}-{sha}
  namespace: {namespace}
  labels:
    created-by: torchx-{version}
spec:
  output:
    to:
{output}
  strategy:
    type: Docker
    dockerStrategy:
      buildArgs:
{build_args}
{image_secret}
  source:
    type: Binary
"""
