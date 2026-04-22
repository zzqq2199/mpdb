# Copyright (c) 2007-2016 Godefroid Chapelle and ipdb development team
#
# This file is part of ipdb.
# Redistributable under the revised BSD license
# https://opensource.org/licenses/BSD-3-Clause

from .__main__ import set_trace, post_mortem, pm, run, iex        # noqa
from .__main__ import runcall, runeval, launch_ipdb_on_exception  # noqa

from .stdout import sset_trace, spost_mortem, spm                 # noqa
from .stdout import slaunch_ipdb_on_exception                     # noqa

from . import web_pdb as _web_pdb

# Only set the default description if the user hasn't set it already.
_desc = _web_pdb._get_description()
if _desc.get("title") is None and _desc.get("subtitle") is None:
    _web_pdb.set_description(
        title="MPDB Web Debugger",
        subtitle='Official site: <a href="https://github.com/zzqq2199/mpdb">https://github.com/zzqq2199/mpdb</a>',
    )
