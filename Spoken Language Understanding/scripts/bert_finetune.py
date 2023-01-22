#coding=utf8
import json
import sys, os, time, gc
from torch.optim import Adam
from transformers import AdamW, get_scheduler

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

from transformers import (
    BertConfig,
    BertTokenizer,
    AutoConfig,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint, is_main_process
from utils.args import init_args
from utils.initialization import *
from utils.example import Example_Bert_manual
from utils.batch import from_example_list_Bert,from_example_list_predict
from utils.vocab import PAD
from model.slu_bert import BertForSlu
# initialization params, output path, logger, random seed and torch.device
args = init_args(sys.argv[1:])
set_random_seed(args.seed)
device = set_torch_device(args.device)
print("Initialization finished ...")
print("Random seed is set to %d" % (args.seed))
print("Use GPU with index %s" % (args.device) if args.device >= 0 else "Use CPU as target torch device")

start_time = time.time()
train_path = os.path.join(args.dataroot, 'train.json')
dev_path = os.path.join(args.dataroot, 'development.json')
Example_Bert_manual.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
#Example_Bert.configuration(args.dataroot, train_path=train_path, word2vec_path=args.word2vec_path)
train_dataset = Example_Bert_manual.load_dataset(train_path)
dev_dataset = Example_Bert_manual.load_dataset(dev_path)
print("Load dataset and database finished, cost %.4fs ..." % (time.time() - start_time))
print("Dataset size: train -> %d ; dev -> %d" % (len(train_dataset), len(dev_dataset)))

args.vocab_size = Example_Bert_manual.word_vocab.vocab_size
args.pad_idx = Example_Bert_manual.word_vocab[PAD]
args.num_tags = Example_Bert_manual.label_vocab.num_tags
args.tag_pad_idx = Example_Bert_manual.label_vocab.convert_tag_to_idx(PAD)

args.lr = 2e-5
args.max_epoch = 21
num_labels = args.num_tags

config = AutoConfig.from_pretrained(
    'bert-base-chinese',
    num_labels=num_labels,
)


model = BertForSlu.from_pretrained('bert-base-chinese',config=config, args=args)

if args.testing:
    check_point = torch.load(open('model.bin','rb'),map_location=device)
    model.load_state_dict(check_point['model'])
    print("Load saved model from root path")

def set_optimizer(model, args):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grouped_params = [{'params': list(set([p for n, p in params]))}]
    optimizer = Adam(grouped_params, lr=args.lr)
    return optimizer


def decode(choice):
    assert choice in ['train', 'dev']
    model.eval()
    dataset = train_dataset if choice == 'train' else dev_dataset
    predictions, labels = [], []
    total_loss, count = 0, 0
    with torch.no_grad():
        for i in range(0, len(dataset), args.batch_size):
            cur_dataset = dataset[i: i + args.batch_size]
            current_batch = from_example_list_Bert(args, cur_dataset, device, train=True)
            pred, label, loss = model.decode(Example_Bert_manual.label_vocab, current_batch)
            for j in range(len(current_batch)):
                if any([l.split('-')[-1] not in current_batch.utt[j] for l in pred[j]]):
                    print(current_batch.utt[j], pred[j], label[j])
            predictions.extend(pred)
            labels.extend(label)
            total_loss += loss
            count += 1
        #print(f"prediction:{predictions}")
        #print(f"labels:{labels}")
        metrics = Example_Bert_manual.evaluator.acc(predictions, labels)
    torch.cuda.empty_cache()
    gc.collect()
    return metrics, total_loss / count

def predict():
    model.eval()
    test_path = os.path.join(args.dataroot, 'test_unlabelled.json')
    test_dataset = Example_Bert_manual.load_dataset(test_path)
    predictions = []
    with torch.no_grad():
        for i in range(0, len(test_dataset), args.batch_size):
            cur_dataset = test_dataset[i:i + args.batch_size]
            #print(cur_dataset[0].utt)
            current_batch = from_example_list_predict(args, cur_dataset, device, train=False)
            print(current_batch[0].utt)
            pred = model.decode(Example_Bert_manual.label_vocab, current_batch)
            predictions += pred
    test_json = json.load(open(test_path,'r'))
    ptr = 0
    for example in test_json:
        for utt in example:
            pred = predictions[ptr]
            print(pred)
            for i in pred:
                utt['pred'].append(i.split('-'))
            ptr += 1

    json.dump(test_json, open(os.path.join(args.dataroot,'prediction.json'),'w'),indent=4,ensure_ascii=False)


if not args.testing:
    num_training_steps = ((len(train_dataset) + args.batch_size - 1) // args.batch_size) * args.max_epoch
    print('Total training steps: %d' % (num_training_steps))
    optimizer = AdamW(model.parameters(), lr=args.lr) #Bert
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    nsamples, best_result = len(train_dataset), {'dev_acc': 0., 'dev_f1': 0.}
    train_index, step_size = np.arange(nsamples), args.batch_size
    print('Start training ......')
    for i in range(args.max_epoch):
        start_time = time.time()
        epoch_loss = 0
        np.random.shuffle(train_index)
        model.train()
        count = 0
        for j in range(0, nsamples, step_size):
            cur_dataset = [train_dataset[k] for k in train_index[j: j + step_size]]
            current_batch = from_example_list_Bert(args, cur_dataset, device, train=True)
            input_ids = current_batch.input_ids.to(device)
            attention_mask = current_batch.attention_mask.to(device)
            token_type_ids = current_batch.token_type_ids.to(device)
            labels = current_batch.tag_ids.to(device)
            tag_mask = current_batch.tag_mask.to(device)
            output, loss = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, labels=labels, tag_mask=tag_mask)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            count += 1
        print('Training: \tEpoch: %d\tTime: %.4f\tTraining Loss: %.4f' % (i, time.time() - start_time, epoch_loss / count))
        torch.cuda.empty_cache()
        gc.collect()

        start_time = time.time()
        metrics, dev_loss = decode('dev')
        dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
        print('Evaluation: \tEpoch: %d\tTime: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, time.time() - start_time, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
        if dev_acc > best_result['dev_acc']:
            best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1'], best_result['iter'] = dev_loss, dev_acc, dev_fscore, i
            torch.save({
                'epoch': i, 'model': model.state_dict(),
                'optim': optimizer.state_dict(),
            }, open('model.bin', 'wb'))
            print('NEW BEST MODEL: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)' % (i, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))

    print('FINAL BEST RESULT: \tEpoch: %d\tDev loss: %.4f\tDev acc: %.4f\tDev fscore(p/r/f): (%.4f/%.4f/%.4f)' % (best_result['iter'], best_result['dev_loss'], best_result['dev_acc'], best_result['dev_f1']['precision'], best_result['dev_f1']['recall'], best_result['dev_f1']['fscore']))
else:
    start_time = time.time()
    metrics, dev_loss = decode('dev')
    dev_acc, dev_fscore = metrics['acc'], metrics['fscore']
    predict()
    print("Evaluation costs %.2fs ; Dev loss: %.4f\tDev acc: %.2f\tDev fscore(p/r/f): (%.2f/%.2f/%.2f)" % (time.time() - start_time, dev_loss, dev_acc, dev_fscore['precision'], dev_fscore['recall'], dev_fscore['fscore']))
