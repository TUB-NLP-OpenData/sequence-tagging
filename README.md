# sequence-tagging
### setup
    pip install -r requirements
    python -m spacy download en_core_web_sm
    
### data

#### scierc-data
    python -c "from util.data_io import download_data; download_data('http://nlp.cs.washington.edu/sciIE/data','sciERC_processed.tar.gz','data',unzip_it=True)"

#### JNLPBA
    git clone https://github.com/allenai/scibert.git
see `scibert/data/ner/JNLPBA`   

### learning-curve on scierc-data

![image](images/learning_curve.png)