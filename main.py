from imagesegmentation import DeepLabV3
import os


if __name__ == "__main__":
    DeepLabV3Obj = DeepLabV3(DATA_ROOT="./input/kaggle_3m/",EPOCHS= 100)

    DeepLabV3Obj.LoadImageData()   # Initialize data set to fit model
    DeepLabV3Obj.TestTrainSplit(size=0.2)  # Splitting data set into 80-20 ration for training and testing respectively
    TrainFlag = True

    if os.path.exists("./output/compiled/DeepLabV3.hd5 "):
        try:
            print("Compiled model is already present and getting loaded")
            DeepLabV3Obj.ModelLoad() # Loading esisting model
            TrainFlag = False
        except Exception as e:
            print("Error while loading trained model")
            TrainFlag = True
    if TrainFlag:
        print("Model fitting is in progress.....")
        DeepLabV3Obj.ModelInitialization() # initializing DeepLabV3 Model
        DeepLabV3Obj.ModelTraining()   # Training model based on the EPOCHS during initialization
    #
    # DeepLabV3Obj.ModelTest() # Testing fitted model for accuracy
    # DeepLabV3Obj.ModelMetrics() # Generating plots for the accuracy