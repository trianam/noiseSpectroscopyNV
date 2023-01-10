#!/usr/bin/env python

import sys
import os
import configurations

if len(sys.argv) < 2 or len(sys.argv) > 2:
    print("Use {} configName".format(sys.argv[0]))
else:
    # conf = getattr(sys.modules['configurations'], sys.argv[1])
    conf = eval('configurations.{}'.format(sys.argv[1]))
    print("====================")
    print("RUN USING {}".format(sys.argv[1]))
    print("tensorboard --logdir=files/{}/tensorBoard".format(conf.path))
    print("====================")
    os.system(" tensorboard --logdir=files/{}/tensorBoard".format(conf.path))