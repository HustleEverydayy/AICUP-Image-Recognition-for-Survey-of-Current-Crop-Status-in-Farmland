# AICUP-Image-Recognition-for-Survey-of-Current-Crop-Status-in-Farmland
近年來民生、工業、醫療等都有完整的AI資料集，但在農業方面數據相對缺乏，故在未來AI智慧農業需求上，將需要投入大量的專業人力進行農業數據蒐集及分析作業。本專題將運用官方提供的圖片，利用深度學習(CNN神經網路)和集成學習去訓練模型，集成學習可以結合眾多的模型產生一個更強大的模型，讓我們可以精確地去影像辨識農作物的類別。

#作業系統:	Windows 10

語言:	Python 3.7

套件:	Standard Library

        os
	
        glob
	
        time
	
        pathlib
	
	Third party library
	      
        Pillow
	
        numpy
	
        PyTorch
	
        Matplotlib
	
預訓練模型:	 InceptionV3
            EfficientNetB0
	    
額外資料集	無

Step-1 用Crop_inceptionV3切圖時，記得將fbl_pt2-E61.pth放在同一個資料夾下，並執行。

Step-2 首先B0_train_program.py要跟train_images放同一個資料夾

	  然後model.load_state_dict(torch.load("modelfile"))
	  modelfile放入你之前訓練的模型，繼續做訓練
	  train_data = datasets.ImageFolder("imgpath", train_transform)
	  imgpath要放你train_images路徑
	  torch.save(model.state_dict("放入你這次訓練完存取的名字 "fbl_en0e60.pth"), outputfile)

Step-3 把fbl_en0e60.pth丟進PyTorch_Prediction_en.py的資料夾並把他改成model.load_state_dict(torch.load("fbl_en0e60.pth")) ，並用cmd去執行，指令如下:

      python PyTorch_Prediction_en.py -d test_images -m fbl_en0e60.pth.pth -o b0_output.csv -b 200
                     檔名                 測試資料集       訓練模型名字          輸出csv檔      速度調整

Step-4 首先將(secret_labels)、(Test20000.csv)、(b0_output.csv)一起丟到(Farmland_Submit1.ipynb)裡面去執行

      再來df_predict = pd.read_csv('輸入你要繳交的csv檔名 這邊我是 resultfirst.csv',header=None, names=colnames)
      之後將resultfirst.csv 下載起來，就是我們上傳到AIcup的csv檔了!!!   


