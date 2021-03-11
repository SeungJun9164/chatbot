#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
import itertools
import math
from torch.jit import script, trace
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
print(torch.__version__)

import tensorflow as tf
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split

corpus_name = 'cornell_movie_dialogs_corpus'
corpus = os.path.join('/home/dilab/tmp/', corpus_name)

def loadLines(fileName, fields):
    lines = {}
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            lineObj = {}
            for i, field in enumerate(fields):
                lineObj[field] = values[i]
                # {'lineID': 'L1045', 'characterID': 'u0', 'movieID': 'm0', 'character': 'BIANCA', 'text': 'They do not!\n'}튜플형식 저장
            lines[lineObj['lineID']] = lineObj 
            # {'L1045': {'lineID': 'L1045', 'characterID': 'u0', 'movieID': 'm0', character': 'BIANCA', 'text': 'They do not!\n'}
            # 튜플 형식으로 저장
    return lines


def loadConversations(fileName, lines, fields):
    conversations = []
    with open(fileName, 'r', encoding='iso-8859-1') as f:
        for line in f:
            values = line.split(" +++$+++ ")
            convObj = {}
            for i, field in enumerate(fields):
                convObj[field] = values[i]
                # {'character1ID': 'u0', 'character2ID': 'u2', 'movieID': 'm0', 'utteranceIDs': "['L194', 'L195', 'L196', 'L197']\n"}
            utterance_id_pattern = re.compile('L[0-9]+')
            lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"]) # ['L194', 'L195', 'L196', 'L197']
            convObj["lines"] = []
            for lineId in lineIds:
                convObj["lines"].append(lines[lineId])
                # lineId에 맞는 lines저장
            conversations.append(convObj)
            # {'character1ID': 'u0', 'character2ID': 'u2', 'movieID': 'm0', 'utteranceIDs': "['L194', 'L195', 'L196', 'L197']\n", 
            # 'lines': [{'lineID': 'L194', 'characterID': 'u0', 'movieID': 'm0', 'character': 'BIANCA', 
            # 'text': 'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'}
    return conversations



def extractSentencePairs(conversations):
    qa_pairs = []
    for conversation in conversations:
        for i in range(len(conversation["lines"]) - 1): # len : L194, L195, L196, L197 -> 4
            # inputLine과 targetLine으로 나눈 이유는 총4개의 대화에서 3번째 대화에 대해 마지막 4번째가 응답이기 때문에
            # 마지막 4번째는 응답할게 없으므로
            # 대화의 마지막 대사는 (그에 대한 응답이 없으므로) 무시합니다
            # 결과적으로 i0 -> i1, i1->i2, i2->i3, i3->i4
            inputLine = conversation["lines"][i]["text"].strip()
            targetLine = conversation["lines"][i+1]["text"].strip()
            
            # 잘못된 샘플은 제거(리스트가 하나라도 비어 있는 경우)
            if inputLine and targetLine:
                qa_pairs.append([inputLine, '[SEP]', targetLine])
    return qa_pairs

# 새로운 파일에 따로 저장
datafile = os.path.join(corpus, "bert_formatted_movie_lines.txt")

delimiter = ' '
# 구분자에 대해 unescape 함수를 호출합니다
delimiter = str(codecs.decode(delimiter, "unicode_escape"))

lines = {}
conversations = []
MOVIE_LINES_FIELDS = ["lineID", "characterID", "movieID", "character", "text"]
MOVIE_CONVERSATIONS_FIELDS = ["character1ID", "character2ID", "movieID", "utteranceIDs"]


lines = loadLines(os.path.join(corpus, "movie_lines.txt"), MOVIE_LINES_FIELDS)
conversations = loadConversations(os.path.join(corpus, "movie_conversations.txt"),
                                  lines, MOVIE_CONVERSATIONS_FIELDS)

# 결과를 새로운 csv 파일로 저장합니다
with open(datafile, 'w', encoding='utf-8') as outputfile:
    writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
    for pair in extractSentencePairs(conversations):
        writer.writerow(pair)

PAD_token = 0
SOS_token = 1
EOS_token = 2
CLS_token = 101
SEP_token = 102
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased', do_lower_case=False)

class Voc:
    def __init__(self, name):
        self.name = name
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {PAD_token : 'PAD', SOS_token:'SOS', EOS_token:'EOS', CLS_token : 'CLS', SEP_token : 'SEP'}
        self.tokenizer = tokenizer
        self.tokens = []
        
    def addSentence(self, sentence): # sentence : 잘 마실게
        sentence = self.tokenizer.encode(sentence)
        tokens = self.tokenizer.convert_ids_to_tokens(sentence)
        self.tokens = tokens

        return tokens            

MAX_LENGTH = 10  

# 유니코드 문자열을 아스키로 변환합니다
# https://stackoverflow.com/a/518232/2809427 참고
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 소문자로 만들고, 공백을 넣고, 알파벳 외의 글자를 제거합니다
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile, corpus_name): # corpus_name : chatData / datafile : formatted_ko_conversations.txt
    lines = open(datafile, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.strip().split('[SEP]')] for l in lines]
    voc = Voc(corpus_name)
    return voc, pairs # voc : 문서 단어집합 / pairs : 문장 쌍 집합

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus, corpus_name, datafile, save_dir):
    voc, pairs = readVocs(datafile, corpus_name) # voc : 단어집합, pairs : 질문 쌍
    pairs = filterPairs(pairs)
    for pair in pairs:
        sentence1 = voc.addSentence(pair[0].strip())
        #print(sentence1)
        sentence2 = voc.addSentence(pair[1].strip())
    return voc, pairs

save_dir = os.path.join("data", "save")
voc, pairs = loadPrepareData(corpus, corpus_name, datafile, save_dir)

def indexesFromSentence(voc, sentence):
    #tokens = tokenizer.tokenize(sentence)
    return tokenizer.encode(sentence)

def zeroPadding(l, fillvalue=PAD_token):
    # print(list(itertools.zip_longest(*l, fillvalue=fillvalue))) : (1957, 2, 2, 2, 0)
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binaryMatrix(l, value=PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def inputVar(l, voc): # l : 응. 새벽에도 일찍 나갔어. 온다간다 말도없이.
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l] 
    # print(indexes_batch) : [[5027, 1239, 9433, 2], [5951, 4686, 1476, 2], [1116, 5309, 2], [319, 2], [186, 2]]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch]) 
    # print(lengths) : tensor([4, 4, 3, 2, 2])
    padList = zeroPadding(indexes_batch)
    # print(padList) : [[5027, 1239, 9433, 2], [5951, 4686, 1476, 2], [1116, 5309, 2, 0], [319, 2, 0, 0], [186, 2, 0, 0]]
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def outputVar(l, voc):
    indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
    #print(indexes_batch)
    max_target_len = max([len(indexes) for indexes in indexes_batch]) 
    padList = zeroPadding(indexes_batch)
    mask = binaryMatrix(padList) 
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2TrainData(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(tokenizer.tokenize(x[0])), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = inputVar(input_batch, voc)
    output, mask, max_target_len = outputVar(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

small_batch_size = 5
batches = batch2TrainData(voc, [random.choice(pairs) for _ in range(small_batch_size)])
input_variable, lengths, target_variable, mask, max_target_len = batches

# print("input_variable:", input_variable)
# print("lengths:", lengths)
# print("target_variable:", target_variable)
# print("mask:", mask)
# print("max_target_len:", max_target_len)

class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding, n_layers=1, dropout=0):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, 
                          dropout = (0 if n_layers == 1 else dropout), bidirectional=True)
        
    def forward(self, input_seq, input_lengths, hidden=None):
        embedded = self.embedding(input_seq) # input_seq : shape=(max_length, batch_size), input_variable
        # print(input_lengths) : =lengths
        # print(embedded.shape) : torch.Size([10, 64, 500]) [max_length, batch_size, hidden_size(은닉상태 크기)]
        
        # nn.utils.rnn.pack_padded_sequence : 패딩연산처리 쉽게하기 위해 중간에 빈공간 제거(형태 : tensor)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths.cpu()) # input_lengths : shape=(batch_size)
        #print(packed.batch_sizes)
        # print(packed.batch_sizes) : tensor([64, 64, 64, 58, 52, 45, 38, 17,  8,  2])
        
        outputs, hidden = self.gru(packed, hidden) # 입력hidden : shape=(n_layers * num_directions, batch_size, hidden_size)
        # print(outputs.batch_sizes) : tensor([64, 64, 63, 52, 47, 34, 24, 18, 12,  6])
        # print(hidden.shape) : torch.Size([4, 64, 500]) [층 * 양방향이면2 아니면1, batch_size, hidden_size]
        
        # nn.utils.rnn.pad_packed_sequence : 패딩연산이 끝난 것을 다시 원래대로 (형태 : torch)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        #print(outputs.shape)
        # print(outputs.shape)# : torch.Size([10, 64, 1000]) # [max_length, batch_size, hidden_size(양방향으로 진행했으면 *2)]
        
        # 양방향 GRU의 출력을 합산합니다
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # print(outputs.shape) : torch.Size([10, 64, 500])
        
        # hidden : GRU의 최종 은닉 상태
        return outputs, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, 'is not an appropriate attention method.')
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))
            
    # 가중치 계산을 dot-product로 계산
    def dot_score(self, hidden, encoder_output):
        # print(torch.sum(hidden * encoder_output, dim=2).shape) : torch.Size([10, 64]) 10개 생성[max_length, batch_size]
        return torch.sum(hidden * encoder_output, dim=2)
    
    # 
    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        # print(energy.shape) : torch.Size([10, 64, 500]) 10개 생성[max_length, batch_size, hidden_size]
        
        # print(torch.sum(hidden * energy, dim=2).shape) : torch.Size([10, 64]) 10개 생성[max_length, batch_size]
        return torch.sum(hidden * energy, dim=2)
    
    
    def concat_score(self, hidden, encoder_output):
        # cat : 합칠 때 차원은 2차원으로 / expand : 확장
        # Tanh 함수는 함수값을 [-1, 1]로 제한시킴
        # print((hidden.expand(encoder_output.size(0), -1, -1).shape)) : torch.Size([10, 64, 500])
        # print(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2).shape) : torch.Size([10, 64, 1000])
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        # print(energy.shape) : torch.Size([10, 64, 500]) 10개 생성[max_length, batch_size, hidden_size]
        return torch.sum(self.v * energy, dim=2)
    
    def forward(self, hidden, encoder_outputs):
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)
            
        attn_energies = attn_energies.t() # t() : 행과 열을 바꿔서 저장[1, 2, 3], [4, 5, 6] -> [1, 4, 7], [2, 5, 8]
        
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # 참조를 보존해 둡니다
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # 레이어를 정의합니다
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # 주의: 한 단위 시간에 대해 한 단계(단어)만을 수행합니다
        # 현재의 입력 단어에 대한 임베딩을 구합니다   
        #print(input_step)
        embedded = self.embedding(input_step) # input_step : 입력 시퀀스 배치에 대한 한 단위 시간(한 단어). shape=(1, batch_size)
        embedded = self.embedding_dropout(embedded)
        #print(embedded.shape)
        # print(embedded.shape) : torch.Size([1, 64, 500])
        
        # 양방향x
        # last_hidden : GRU의 마지막 은닉 레이어. shape=(n_layers * num_directions, batch_size, hidden_size)
        # print(last_hidden.shape) : torch.Size([2, 64, 500]) 
        rnn_output, hidden = self.gru(embedded, last_hidden) 
        # print(rnn_output.shape) : torch.Size([1, 64, 500])
        # print(hidden.shape) : torch.Size([2, 64, 500])

        # attention 가중치
        attn_weights = self.attn(rnn_output, encoder_outputs) # encoder_outputs : 인코더 모델 출력 shape=(max_length, batch_size, hidden_size)
        # print(attn_weights.shape) : torch.Size([64, 1, 10]) 

        # 인코더 출력에 어텐션을 곱하여 새로운 context vector생성
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # print(context.shape) : torch.Size([64, 1, 500])

        rnn_output = rnn_output.squeeze(0) # print(rnn_output.shape) : torch.Size([64, 500])
        context = context.squeeze(1) # print(context.shape) : torch.Size([64, 500])
        concat_input = torch.cat((rnn_output, context), 1) # print(concat_input.shape) : torch.Size([64, 1000])
        concat_output = torch.tanh(self.concat(concat_input))
        # print(concat_output.shape) : torch.Size([64, 500])

        # output : 각 단어가 디코딩된 시퀀스에서 다음 단어로 사용되었을 때 적합할 확률을 나타내는 정규화된 softmax 텐서. 
        # shape=(batch_size, voc.num_words)
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)

        return output, hidden

def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def train(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding, encoder_optimizer, decoder_optimizer,
         batch_size, clip, max_length = MAX_LENGTH):
    
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)
    
    loss = 0
    print_losses = []
    n_totals = 0
    
    # EncoderRNN의 forward부분 실행
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)
    
    # 초기 디코더 입력을 생성(각 문장을 SOS 토큰으로 시작)
    decoder_input = torch.LongTensor([[CLS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)
    
    # 디코더의 초기 은닉 상태를 인코더의 마지막 은닉 상태로
    decoder_hidden = encoder_hidden[:decoder.n_layers]
    
    # teacher_forcing : Decoder부분에서 앞 단어가 잘못 추측되었을 경우 뒤에도 달라지니 정답을 입력해 주는 것
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
    
    if use_teacher_forcing:
        for t in range(max_target_len):
            # LuongAttnDecoderRNN의 forward로 실행
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
             # Teacher forcing 사용: 다음 입력을 현재의 목표로 둡니다
            decoder_input = target_variable[t].view(1, -1)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
            
            # Teacher forcing 미사용: 다음 입력을 디코더의 출력으로 둡니다
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
            
    loss.backward()
    
    # clip_grad_norm_: 그라디언트를 제자리에서 수정합니다
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return sum(print_losses) / n_totals


def trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size, print_every, save_every, clip, corpus_name, loadFilename):

    # 각 단계에 대한 배치 설정
    # batch2TrainData : return inp, lengths, output, mask, max_target_len
    training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]

    start_iteration = 1
    print_loss = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    print("Training...")

    for iteration in range(start_iteration, n_iteration + 1):
        torch.cuda.empty_cache() # GPU 캐시 데이터 삭제
        training_batch = training_batches[iteration - 1]
        
        input_variable, lengths, target_variable, mask, max_target_len = training_batch


        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        
        print_loss += loss


        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg))
            print_loss = 0

        # Checkpoint를 저장
        if (iteration % save_every == 0):
            directory = os.path.join(save_dir, model_name, corpus_name, 'large', '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,
                'en': encoder.state_dict(),
                'de': decoder.state_dict(),
                'en_opt': encoder_optimizer.state_dict(),
                'de_opt': decoder_optimizer.state_dict(),
                'loss': loss,
                'voc_dict': voc.__dict__,
                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpoint')))


model_name = 'cb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 768
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
batch_size = 64


loadFilename = None
checkpoint_iter = 50000
loadFilename = os.path.join(save_dir, model_name, corpus_name, 'large',
                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
                            '{}_checkpoint.tar'.format(checkpoint_iter))


# loadFilename이 제공되는 경우에는 모델을 불러옵니다
if loadFilename:
    # 모델을 학습할 때와 같은 기기에서 불러오는 경우
    checkpoint = torch.load(loadFilename)
    # GPU에서 학습한 모델을 CPU로 불러오는 경우
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc.__dict__ = checkpoint['voc_dict']

embedding = nn.Embedding(len(tokenizer.vocab), hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(attn_model, embedding, hidden_size, len(tokenizer.vocab), decoder_n_layers, dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

encoder = encoder.to(device)
decoder = decoder.to(device)

clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 50000
print_every = 1
save_every = 10000

# Dropout 레이어를 학습 모드로 둡니다
encoder.train()
decoder.train()

# Optimizer를 초기화합니다
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)


for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# 학습 단계를 수행합니다
trainIters(model_name, voc, pairs, encoder, decoder, encoder_optimizer, decoder_optimizer,
           embedding, encoder_n_layers, decoder_n_layers, save_dir, n_iteration, batch_size,
           print_every, save_every, clip, corpus_name, loadFilename)

# 탐욕적 디코딩(Greedy decoding) : 각 단계에 대해 단순히 decoder_output 에서 가장 높은 softmax값을 갖는 단어를 선택하는 방식
class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length):

        # EncoderRNN의 forward부분 실행
        encoder_outputs, encoder_hidden = self.encoder(input_seq, input_length)
        #print('outputs : ', encoder_outputs.shape)
        #print('hidden : ', encoder_hidden.shape)
               
        # encoder의 마지막 hidden이 decoder의 처음 hidden
        decoder_hidden = encoder_hidden[:decoder.n_layers]
        #print('decoder_hidden : ', decoder_hidden.shape)
        
        # decoder의 처음입력을 SOS로 초기화
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * CLS_token
        #print('decoder_input : ', decoder_input)
        
        # 디코더가 단어를 덧붙여 나갈 텐서를 초기화
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)

        for _ in range(max_length):
            # LuongAttnDecoderRNN의 forward로 실행
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
            #print('decoder_output : ', decoder_output.shape)
            #print('decoder_hidden : ', decoder_hidden.shape)
            
            # 가장 가능성 높은 단어 토큰과 그 softmax 점수를 구합니다
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            #print('decoder_scores : ', decoder_scores)
            #print('decoder_input : ', decoder_input)

            # 토큰, 점수 기록
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)

            # 현재의 토큰을 디코더의 다음 입력으로 준비시킵니다(차원을 증가시켜서)
            decoder_input = torch.unsqueeze(decoder_input, 0)

        return all_tokens, all_scores

def evaluate(encoder, decoder, searcher, voc, sentence, max_length=MAX_LENGTH):
    # indexes_batch : 문장을 단어집합에 저장된 수로 바꾼후 마지막에 EOS추가하는 함수
    indexes_batch = [tokenizer.encode(sentence)]
    #print('input : ', indexes_batch)
    
    # lengths 텐서를 만듭니다
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    #print(lengths)
    
    # 배치의 차원을 뒤집어서 모델이 사용하는 형태로 만듭니다
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    #print('input : ', input_batch)
    
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    
    # searcher를 이용하여 문장을 디코딩합니다
    G_tokens, scores = searcher(input_batch, lengths, max_length)
    #print(G_tokens)
    #print(scores)

    # 인덱스 -> 단어    
    decoded_words = tokenizer.convert_ids_to_tokens(G_tokens) ###수정해야함
    #print(decoded_words)
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        result = []
        temp = []
        cnt = 0
        try:
            # 입력 문장을 받아옵니다
            input_sentence = input('> ')
            # 종료 조건인지 검사합니다
            if input_sentence == 'exit' or input_sentence == 'quit': break
            # 문장을 평가합니다
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # 응답 문장을 형식에 맞춰 출력합니다
            output_words[:] = [x for x in output_words if not (x == '[CLS]' or x == '[SEP]' or x == 'PAD' or x == 'SOS' or x == 'EOS')]
            
            for i in output_words:
                if i == '.' or i == '?':
                    temp.append(i)
                    break
                else:
                    temp.append(i)

	    # ##제거
            for i, text in enumerate(temp):
                if text[0] == '#':
                    cnt += 1
                    temp = result[i - cnt] + output_words[output_words.index(text)].replace(text, text[2:])
                    result.append(temp)
                    result.pop(i - cnt)
                else:
                    result.append(text)
            print('Bot:', ' '.join(result))

        except KeyError:
            print("Error: Encountered unknown word.")

# Dropout 레이어를 평가 모드로 설정합니다
encoder.eval()
decoder.eval()

# 탐색 모듈을 초기화합니다
searcher = GreedySearchDecoder(encoder, decoder)

# 채팅을 시작합니다 (다음 줄의 주석을 제거하면 시작해볼 수 있습니다)
evaluateInput(encoder, decoder, searcher, voc)

