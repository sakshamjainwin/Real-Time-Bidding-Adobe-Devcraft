"""
This script generates pCTR (predicted Click-Through Rate) values for train and test datasets using a pre-trained CTR model.
It loads a pre-trained CTR model, processes the train and test data in batches, and generates pCTR values.
Then creates dataframes for train and test sets with click, pctr, market_price, minutes and 24_time_fraction.
Finally, it saves the data in different formats for different bidding strategies (LIN, RLB, DRLB).
"""
import os
import time
import utils.config as config
import pandas as pd
import torch

from tqdm import tqdm
from main import get_model, get_dataset

if __name__ == '__main__':
    start_time = time.time()
    args = config.init_parser()
    train_data, val_data, test_data, field_nums, feature_nums = get_dataset(args)
    device = torch.device(args.device)

    ctr_model = get_model(args.ctr_model, feature_nums, field_nums, args.latent_dims).to(device)
    pretrain_params = torch.load(
        os.path.join(args.save_param_dir, args.campaign_id, args.ctr_model + 'best.pth'))
    ctr_model.load_state_dict(pretrain_params)

    train_ctrs = []
    test_ctrs = []

    counter = 0
    shape = train_data.shape[0]
    iter = int(shape / 1024)
    # Iterate through the training data in batches of 1024
    for batch in tqdm(range(iter)):
        counter = batch
        # Generate pCTR for the current batch
        _ = ctr_model(
            torch.LongTensor(train_data[batch * 1024:(batch + 1) * 1024, 5:].astype(int)).to(
                args.device)).detach().cpu().numpy()
        train_ctrs.extend(_.flatten().tolist())

    # Generate pCTR for the remaining training data
    tail = ctr_model(
        torch.LongTensor(train_data[(counter + 1) * 1024:, 5:].astype(int)).to(args.device)).detach().cpu().numpy()
    train_ctrs.extend(tail.flatten().tolist())

    counter = 0
    shape = test_data.shape[0]
    iter = int(shape / 1024)
    # Iterate through the test data in batches of 1024
    for batch in tqdm(range(iter)):
        counter = batch
        # Generate pCTR for the current batch
        _ = ctr_model(
            torch.LongTensor(test_data[batch * 1024:(batch + 1) * 1024, 5:].astype(int)).to(
                args.device)).detach().cpu().numpy()
        test_ctrs.extend(_.flatten().tolist())

    # Generate pCTR for the remaining test data
    tail = ctr_model(
        torch.LongTensor(test_data[(counter + 1) * 1024:, 5:].astype(int)).to(args.device)).detach().cpu().numpy()
    test_ctrs.extend(tail.flatten().tolist())

    # Create dictionaries for train and test data
    train = {'clk': train_data[:, 0].tolist(),  # Click labels
             'pctr': train_ctrs,  # Predicted CTRs
             'market_price': train_data[:, 1].tolist(),  # Market prices
             'minutes': train_data[:, 4].tolist(), # Minutes
             '24_time_fraction': train_data[:, 2].tolist(),  # 24-hour time fraction
             }

    test = {'clk': test_data[:, 0].tolist(),
            'pctr': test_ctrs,
            'market_price': test_data[:, 1].tolist(),
            'minutes': test_data[:, 4].tolist(),
            '24_time_fraction': test_data[:, 2].tolist(),
            }
    # Save the data in different formats for different bidding strategies
    save_path = os.path.join(args.data_path, args.campaign_id, args.predictions_path)
    
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    
    train_df = pd.DataFrame(data=train)
    test_df = pd.DataFrame(data=test)

    # Generate additional time features for LIN and DRLB data
    train_df['48_time_fraction'] = train_df['24_time_fraction'] * 2 + (
        train_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 30)))
    test_df['48_time_fraction'] = test_df['24_time_fraction'] * 2 + (
        test_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 30)))
    train_df['96_time_fraction'] = train_df['24_time_fraction'] * 4 + (
        train_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))
    test_df['96_time_fraction'] = test_df['24_time_fraction'] * 4 + (
        test_df['minutes'].apply(lambda x: int(int(str(x)[10:12]) / 15)))
    train_df['day'] = train_df.minutes.apply(lambda x: int(str(x)[6:8]))
    test_df['day'] = test_df.minutes.apply(lambda x: int(str(x)[6:8]))

    # Save train and test data for LIN bidding strategy
    train_df.to_csv(os.path.join(save_path, 'train.bid.lin.csv'), index=None)
    test_df.to_csv(os.path.join(save_path, 'test.bid.lin.csv'), index=None)

    # Generate and save data for RLB bidding strategy
    # origin_data = test_df
    # rlb_data = origin_data[['clk', 'pctr', 'market_price']]
    # fout = open(os.path.join(save_path, 'test.bid.rlb.txt'), 'w')
    # for index, row in rlb_data.iterrows():
    #     fout.write(str(int(row['clk'])) + " " + str(int(row['market_price'])) + " " + str(row['pctr']) + '\n')
    # fout.close()

    end_time = time.time()
    print('Time cost: ', end_time - start_time)