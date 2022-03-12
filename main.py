import argparse
import os
from model import Model
import tensorflow.contrib as gan
parser = argparse.ArgumentParser()


parser.add_argument("--lr",type=float,help="learning_rate",default=0.0002)

parser.add_argument("--batch_size",type=int,help="batch size",default=16)

parser.add_argument("--input_size",type=int,help="image input size ",default=256)

parser.add_argument("--output_size",type=int,help="image output size",default=64)

parser.add_argument("--epoch",type=int,help="number of epochs",default=1000)

parser.add_argument("--step",type=int,help="how many roop in a epoch",default=200)
parser.add_argument("--mode",type=str,help="select mode training or test",default='train')
parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint/', help='models are saved here')
parser.add_argument('--loss', type=str ,default='vanila')
parser.add_argument('--ckpt', type=str , default=None)
parser.add_argument('--test_data', type=str , default='test/original')
parser.add_argument('--result_dir', type=str , default='result')
parser.add_argument('--train_root', type=str , default='train')




parser.add_argument('--test_dir', type=str,default='test')

args=parser.parse_args()

if not os.path.exists(args.checkpoint_dir):
    os.makedirs(args.checkpoint_dir)


if args.mode == 'train':
    model = Model(output_size=args.output_size,
                  input_size=args.input_size,
                  epoch=args.epoch,
                  step=args.step,
                  lr=args.lr,
                  batch_size=args.batch_size,
                  checkpoint_dir=args.checkpoint_dir,
                  mode=args.mode,
                  loss=args.loss,
                  train_root=args.train_root
                  )
    model.main()
elif args.mode =='test':
    model = Model(output_size=args.output_size,
                  input_size=args.input_size,
                  epoch=args.epoch,
                  step=args.step,
                  lr=args.lr,
                  batch_size=args.batch_size,
                  checkpoint_dir=args.checkpoint_dir,
                  mode=args.mode,
                  loss=args.loss)
    # model.test(args.test_dir,args.ckpt,args.test_data,args.result_dir)
    model.test(args.ckpt,args.test_data,args.result_dir)
