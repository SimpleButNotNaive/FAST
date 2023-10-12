import pickle
from meta import MetaLearner
from options import args
from model_training import test, fair_dynamic_train, set_seed

data_dir = "data_processed"
model_dir = 'models'


train_total_xs = pickle.load(open(f"./{data_dir}/train_total_xs", 'rb'))
train_total_ys = pickle.load(open(f"./{data_dir}/train_total_ys", 'rb'))
train_genders = pickle.load(open(f"./{data_dir}/train_gender", 'rb'))

test_total_xs = pickle.load(open(f"./{data_dir}/test_total_xs", 'rb'))
test_total_ys = pickle.load(open(f"./{data_dir}/test_total_ys", 'rb'))
test_genders = pickle.load(open(f"./{data_dir}/test_gender", 'rb'))

mae_l = []
ndcg_l = []
mse_l = []

mae_gap_l = []
ndcg_gap_l = []
mse_gap_l = []

seed = 1024

if args.model == 'Ours':

    set_seed(seed)
    meta = MetaLearner(args)
    fair_dynamic_train(meta, 
        train_total_xs, train_total_ys, train_genders, 
        args.gamma, args.num_epoch, args.batch_size, args.tau, args.device)
    mae, mse, mae_gap, mse_gap = test(meta, test_total_xs, test_total_ys, test_genders, args.device)

    print(f"MAE loss: {mae:.3f}")
    print(f"MSE loss: {mse:.3f}")

    print(f"MAE gap: {mae_gap:.3f}")
    print(f"MSE gap: {mse_gap:.3f}")
