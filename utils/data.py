"""
This script provides functions for:
    1. Setting up a random seed for reproducibility.
    2. Converting time into fractions.
    3. Converting data into the libsvm format.
    4. Converting original log data into csv format.
"""
import csv
import operator
import argparse
import random
import numpy as np
import os
import config

def setup_seed(seed):
    """
    Sets up the random seed for numpy and random libraries.

    Args:
        seed (int): The seed value.
    """
    np.random.seed(seed)
    random.seed(seed)

def to_time_frac(hour, min, time_frac_dict):
    """
    Converts the given hour and minute into a time fraction based on the provided dictionary.

    Args:
        hour (int): The hour.
        min (int): The minute.
        time_frac_dict (dict): A dictionary mapping hours to minute ranges and their corresponding fraction indices.

    Returns:
        str: The time fraction index as a string.
    """
    for key in time_frac_dict[hour].keys():
        if key[0] <= min <= key[1]:
            return str(time_frac_dict[hour][key])

def to_libsvm_encode(datapath, time_frac_dict, global_featindex=None, maxindex=0):
    """
    Converts the bid data to libsvm format using global feature indexing.
    
    Args:
        datapath (str): Path to campaign directory
        time_frac_dict (dict): Time fraction mapping
        global_featindex (dict): Global feature index mapping
        maxindex (int): Current maximum feature index
    
    Returns:
        tuple: Updated global feature index and max index
    """
    print(f'Processing campaign in: {datapath}')
    
    if global_featindex is None:
        global_featindex = {'truncate': 0}
        maxindex = 1

    oses = ["windows", "ios", "mac", "android", "linux"]
    browsers = ["chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie"]

    f1s = ["weekday", "hour", "IP", "region", "city", "adexchange", "domain", "slotid", "slotwidth", "slotheight",
           "slotvisibility", "slotformat", "creative", "advertiser"]

    f1sp = ["useragent", "slotprice"]

    def feat_trans(name, content):
        """
        Transforms features based on their name and content.

        Args:
            name (str): The name of the feature.
            content (str): The content of the feature.

        Returns:
            str: The transformed feature.
        """
        # Feature transformation
        content = content.lower()
        # Operating system and browser
        if name == "useragent":
            operation = "other"
            for o in oses:
                if o in content:
                    operation = o
                    break
            browser = "other"
            for b in browsers:
                if b in content:
                    browser = b
                    break
            return operation + "_" + browser
        # Floor price
        if name == "slotprice":
            price = int(content)
            if price > 100:
                return "101+"
            elif price > 50:
                return "51-100"
            elif price > 10:
                return "11-50"
            elif price > 0:
                return "1-10"
            else:
                return "0"

    def getTags(content):
        """
        Extracts and returns a list of usertags from the given content.

        Args:
            content: usertag content string

        Returns:
            A list of usertags
        """
        if content == '\n' or len(content) == 0:
            return ["null"]
        return content.strip().split(',')[:5]
    
    namecol = {}
    fi = open(os.path.join(datapath, 'train.bid.csv'), 'r')
    
    first = True
    for line in fi:
        s = line.split(',')
        if first:
            first = False
            for i in range(0, len(s)):
                namecol[s[i].strip()] = i
                feat = str(i) + ':other'
                if feat not in global_featindex:
                    global_featindex[feat] = maxindex
                    maxindex += 1
            continue
            
        # Process features using global_featindex instead of local featindex
        for f in f1s:
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                global_featindex[feat] = maxindex
                maxindex += 1
                
        # Process f1sp features
        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                global_featindex[feat] = maxindex
                maxindex += 1

        # Process usertag
        col = namecol["usertag"]
        tags = getTags(s[col])
        feat = str(col) + ':' + ''.join(tags)
        if feat not in global_featindex:
            global_featindex[feat] = maxindex
            maxindex += 1

    # Write the campaign-specific train and test files using global_featindex
    print('indexing ' + datapath + '/train.bid.csv')
    fi = open(os.path.join(datapath, 'train.bid.csv'), 'r')
    fo = open(os.path.join(datapath, 'train.bid.txt'), 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        # click + winning price + hour + time_frac
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict) + ',' + str(s[4]))
        index = global_featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                feat = str(col) + ':other'
            index = global_featindex[feat]
            fo.write(',' + str(index))

        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                feat = str(col) + ':other'
            index = global_featindex[feat]
            fo.write(',' + str(index))

        # usertag tag trick
        col = namecol["usertag"]
        tags = getTags(s[col])
        feat = str(col) + ':' + ''.join(tags)
        if feat not in global_featindex:
            feat = str(col) + ':other'
        index = global_featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()

    # indexing test
    print('indexing ' + datapath + '/test.bid.csv')
    fi = open(os.path.join(datapath, 'test.bid.csv'), 'r')
    fo = open(os.path.join(datapath, 'test.bid.txt'), 'w')

    first = True
    for line in fi:
        if first:
            first = False
            continue
        s = line.split(',')
        time_frac = s[4][8: 12]
        fo.write(s[0] + ',' + s[23] + ',' + s[2] + ',' + to_time_frac(int(time_frac[0:2]), int(time_frac[2:4]),
                                                                      time_frac_dict) + ',' + str(s[4]))
        index = global_featindex['truncate']
        fo.write(',' + str(index))
        for f in f1s:  # every direct first order feature
            col = namecol[f]
            if col >= len(s):
                print('col: ' + str(col))
                print(line)
            content = s[col]
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                feat = str(col) + ':other'
            index = global_featindex[feat]
            fo.write(',' + str(index))
        for f in f1sp:
            col = namecol[f]
            content = feat_trans(f, s[col])
            feat = str(col) + ':' + content
            if feat not in global_featindex:
                feat = str(col) + ':other'
            index = global_featindex[feat]
            fo.write(',' + str(index))
        col = namecol["usertag"]
        tags = getTags(s[col])
        feat = str(col) + ':' + ''.join(tags)
        if feat not in global_featindex:
            feat = str(col) + ':other'
        index = global_featindex[feat]
        fo.write(',' + str(index))
        fo.write('\n')
    fo.close()
    
    return global_featindex, maxindex

if __name__ == '__main__':
    setup_seed(1)
    args = config.init_parser()
    
    # Initialize time fraction dictionary
    time_frac_dict = {}
    count = 0
    for i in range(24):
        hour_frac_dict = {}
        for item in [(0, 15), (15, 30), (30, 45), (45, 60)]:
            hour_frac_dict.setdefault(item, count)
            count += 1
        time_frac_dict.setdefault(i, hour_frac_dict)

    # Process all campaigns to build global feature index
    global_featindex = None
    maxindex = 0
    
    # First pass: build global feature index
    for campaign_id in os.listdir(args.data_path):
        campaign_path = os.path.join(args.data_path, campaign_id)
        if os.path.isdir(campaign_path):
            if args.is_to_csv:
                # Convert log files to CSV (keep existing CSV conversion code)
                print(f'Converting {campaign_id} to CSV')
                with open(os.path.join(campaign_path, 'train.bid.csv'), 'w', newline='') as csv_file:
                    spam_writer = csv.writer(csv_file, dialect='excel')
                    with open(os.path.join(campaign_path, 'train.log.txt'), 'r') as filein:
                        for line in filein:
                            line_list = line.strip('\n').split('\t')
                            spam_writer.writerow(line_list)

                with open(os.path.join(campaign_path, 'test.bid.csv'), 'w', newline='') as csv_file:
                    spam_writer = csv.writer(csv_file, dialect='excel')
                    with open(os.path.join(campaign_path, 'test.log.txt'), 'r') as filein:
                        for line in filein:
                            line_list = line.strip('\n').split('\t')
                            spam_writer.writerow(line_list)
                            
            global_featindex, maxindex = to_libsvm_encode(campaign_path, time_frac_dict, global_featindex, maxindex)
            
            # Clean up CSV files
            os.remove(os.path.join(campaign_path, 'train.bid.csv'))
            os.remove(os.path.join(campaign_path, 'test.bid.csv'))

    # Write global feature index
    featvalue = sorted(global_featindex.items(), key=operator.itemgetter(1))
    with open(os.path.join(args.data_path, 'feat.bid.txt'), 'w') as fo:
        fo.write(str(maxindex) + '\n')
        for fv in featvalue:
            fo.write(fv[0] + '\t' + str(fv[1]) + '\n')
