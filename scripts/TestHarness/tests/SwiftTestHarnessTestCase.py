#*                    DO NOT MODIFY THIS HEADER
#*             Swift, a Fourier spectral solver for MOOSE
#*
#*            Copyright 2024 Battelle Energy Alliance, LLC
#*                        ALL RIGHTS RESERVED
#*
#*        Licensed under LGPL 2.1, please see LICENSE for details
#*             https://www.gnu.org/licenses/lgpl-2.1.html

import os, sys, subprocess, tempfile

import TestHarness
sys.path.append(os.path.join(TestHarness.__path__[0], 'tests'))
from TestHarnessTestCase import TestHarnessTestCase

class SwiftTestHarnessTestCase(TestHarnessTestCase):
    """
    TestCase class for running TestHarness commands.
    """

    def runTests(self, *args, tmp_output=True):
        cmd = ['./run_tests'] + list(args) + ['--term-format', 'njCst']
        sp_kwargs = {'cwd': os.path.join(os.path.dirname(__file__), '..', '..', '..'),
                     'text': True}
        if tmp_output:
            with tempfile.TemporaryDirectory() as output_dir:
                cmd += ['-o', output_dir]
            return subprocess.check_output(cmd, **sp_kwargs)
        return subprocess.check_output(cmd, **sp_kwargs)

