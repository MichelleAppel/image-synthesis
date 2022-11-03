config = {
    # 'bsds500': {'data_root': 'path_to/bsds500/HED-BSDS/',
    #             'data_lst': 'train_pair.lst',
    #             'mean_bgr': [104.00699, 116.66877, 122.67892],
    #             'yita': 0.5},
    'NYUDv2': {'data_root': r'C:\Users\appel\Documents\Project\simulation-synthesis\output\448\labeled_real',
                   'mean_bgr': [104.00699, 116.66877, 122.67892],
                   'yita': 0.5},
    'ArchVizPro': {'data_root': r'C:\Users\appel\Documents\Project\simulation-synthesis\output\448\labeled_fake',
                   'mean_bgr': [104.00699, 116.66877, 122.67892], #? TODO
                   'yita': 0.5},
}

config_test = {
    # 'bsds500': {'data_root': 'path_to/bsds500/BSR/BSDS500/data/',
    #             'data_lst': 'test_pair.lst',
    #             'mean_bgr': [104.00699, 116.66877, 122.67892],
    #             'yita': 0.5},
    'NYUDv2': {'data_root': r'C:\Users\appel\Documents\Project\simulation-synthesis\output\448\labeled_real_test',
                   'mean_bgr': [104.00699, 116.66877, 122.67892],
                   'yita': 0.5},
    'ArchVizPro': {'data_root': r'C:\Users\appel\Documents\Project\simulation-synthesis\output\448\labeled_fake_test',
                   'mean_bgr': [104.00699, 116.66877, 122.67892], #? TODO
                   'yita': 0.5},
}

if __name__ == '__main__':
    print(config.keys())
