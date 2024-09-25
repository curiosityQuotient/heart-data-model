import os
import shutil
from datetime import datetime

mlrun_dir = "../mlruns/493198603292191274/"
contents = os.listdir(mlrun_dir)
dirs = [mlrun_dir + dd for dd in contents if os.path.isdir(dd) or dd != "datasets"]
dirs.sort(key=os.path.getmtime, reverse=True)

newest_dir = dirs[0]
print("Newest mlrun folder identified as: ", newest_dir)
print("Updated: ", datetime.fromtimestamp(os.stat(newest_dir).st_mtime))

shutil.copy2(
    newest_dir + "/artifacts/model/model.pkl",
    "../latest_model",
)
