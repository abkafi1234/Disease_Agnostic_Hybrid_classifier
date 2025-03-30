For binary classification: 
  1. put the images in the "Binary Classification/Dataset"
  2. Run the Binary Classification.ipynb
  3. you will get the resulting randomforest model as .pkl. then run the Streamlitapp.py with "streamlit run app.py"
For multiclass classificaiton:
  1. put the images in the "Multiclass Classification/Dataset"
  2. Run the Binary Classification.ipynb
  3. you will get the resulting densenet model as .onnx. then run the Multiclassapp.py with "streamlit run Multiclassapp.py"
For Hybrid model
  1. Put the randomforest.pkl and .onnx model in the directory and run "streamlit run Hybridmodel.py".
