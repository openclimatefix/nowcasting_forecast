"""
get the GSP installed solar cpaiacty from pv live

pip install git+https://github.com/SheffieldSolar/PV_Live-API

This script can take ~30 secdson to run
"""

from datetime import datetime

import pytz
from nowcasting_dataset.data_sources.gsp.pvlive import get_installed_capacity

c = get_installed_capacity(start=datetime(2022, 2, 14, tzinfo=pytz.utc))

c.to_csv("./nowcasting_forecast/data/gsp_capacity.csv")
