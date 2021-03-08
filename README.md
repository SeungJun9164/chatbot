# chatbot

### bert_en_pytorch_chatbot_tutorial.py

- [DataSet](#DataSet)\
[Cornell_Movie-Dialogs_Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html)

- [Preparation](#preparation)
```
!pip install transformers
!pip install torch==1.8.0+cu102 torchvision==0.9.0+cu102 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
!pip install sklearn
```

- [How to use](#how-to-use)
  - [model.pt](https://drive.google.com/file/d/1LUSy1yd9MztKPam9H6MLexAwPKqv2Qmb/view?usp=sharing)를 다운
  - bert_en_pytorch_chatbot_tutorial.py 파일의 아래 코드부분을 자신이 원하는 경로로 설정
    ```
    corpus_name = 'cornell_movie_dialogs_corpus' 
    corpus = os.path.join('/home/dilab/tmp/', corpus_name)
    ```
  - [Start](#Start)
    ```python
    >>> python bert_en_pytorch_chatbot_tutorial.py
    ```
- [Result](#Result)\
![image](https://user-images.githubusercontent.com/60804222/110282746-5cda8400-8022-11eb-9ad3-ba7aca4a7719.png)

- [Reference](#Reference)\
[pytorch chatbot tutorial](https://pytorch.org/tutorials/beginner/chatbot_tutorial.html)\
[huggingface](https://huggingface.co/transformers/pretrained_models.html)

