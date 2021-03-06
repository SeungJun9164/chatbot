# chatbot

### bert_large_en_pytorch_chatbot_tutorial.py / bert_model_large_en_pytorch_chatbot_tutorial.py
  - [Difference](#Difference)
    - 1. tokenizer / tokenizer + bert_model
          ```
          tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)
          # bert_model = BertModel.from_pretrained('bert-large-uncased') # Add Encoder outputs
          ```
    - 2. hidden_size
          ```
          hidden_size = 768
          hidden_size = 1024
          ```
- [DataSet](#DataSet)\
[Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

- [Preparation](#preparation)
```
!pip install transformers
!pip install torch==1.8.0+cu102 torchvision==0.9.0+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip install sklearn
```

- [How to use](#how-to-use)
  - [bert_large_model.pt](https://drive.google.com/file/d/1LUSy1yd9MztKPam9H6MLexAwPKqv2Qmb/view?usp=sharing) 다운
  - [bert_model_large_model.pt](https://drive.google.com/file/d/1s9ZW9LJAVeuV88RQI3YV9PNNbkYgEDZN/view?usp=sharing) 다운
  - bert_en_pytorch_chatbot_tutorial.py 파일의 아래 코드부분을 자신이 원하는 경로로 설정
    ```
    corpus_name = 'cornell_movie_dialogs_corpus' 
    corpus = os.path.join('/home/dilab/tmp/', corpus_name)
    
    loadFilename = os.path.join(save_dir, model_name, corpus_name, 'large_model',
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))

    ```
  - [Start](#Start)
    ```python
    >>> python bert_en_pytorch_chatbot_tutorial.py
    ```
- [Result](#Result)\
![image](https://user-images.githubusercontent.com/60804222/110282746-5cda8400-8022-11eb-9ad3-ba7aca4a7719.png)
![image](https://user-images.githubusercontent.com/60804222/110897238-6de80580-8340-11eb-9c4f-117855ca1017.png)

- [Reference](#Reference)\
[pytorch chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)\
[huggingface](https://huggingface.co/transformers/pretrained_models.html)

