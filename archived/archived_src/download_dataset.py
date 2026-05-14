from roboflow import Roboflow
rf = Roboflow(api_key="nFbtFTVxsmROfGelu8Qn")
project = rf.workspace("mubaraks-workspace").project("bottle_inspection-0cpal")
version = project.version(4)
dataset = version.download("yolov5")
                