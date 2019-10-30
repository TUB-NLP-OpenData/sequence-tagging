# sequence-tagging
### setup
    pip install -r requirements
    
### download scierc-data
    python -c "from util.data_io import download_data; download_data('http://nlp.cs.washington.edu/sciIE/data','sciERC_processed.tar.gz','data',unzip_it=True)"
    
### learning-curve on scierc-data

![image](images/learning_curve.png)