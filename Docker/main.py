import Preprocessing as py

import Model_Classification as mc
import Model_Regression as mr

import pandas as pd

def main():
    # print the output heading model results centered
    print("\n")
    print("-----------------------------------------------------------------------")
    print("*********************Regression Model Results**********************")
    print("-----------------------------------------------------------------------")
    print("\n")
    mse_df = mr.regressor()
    print(mse_df)
    print("\n")
    print("\n")
    
    
    print("\n")
    print("-----------------------------------------------------------------------")
    print("*********************Classification Model Results**********************")
    print("-----------------------------------------------------------------------")
    print("\n")
    model_df = mc.classfication()
    print(model_df)
    print("\n")
    print("\n")


if __name__ == "__main__":
    main()
