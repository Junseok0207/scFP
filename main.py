import torch
from misc.utils import set_seed, set_filename, setup_logger
from argument import parse_args
import numpy as np
import datetime

def main():
    args, _ = parse_args()
    torch.set_num_threads(3)
    
    rmse_list, median_l1_distance_list, cosine_similarity_list = [], [], []
    imputed_ari_list, imputed_nmi_list, imputed_ca_list = [], [], []
    reduced_ari_list, reduced_nmi_list, reduced_ca_list = [], [], []
 
    file = set_filename(args)
    logger = setup_logger('./', '-', file)
    for seed in range(0, args.n_runs):
        print(f'Seed: {seed}, Filename: {file}')
        set_seed(seed)
        args.seed = seed

        from models import scFP_Trainer
        embedder = scFP_Trainer(args)

        if (args.drop_rate != 0.0):
            [rmse, median_l1_distance, cosine_similarity, imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca] = embedder.train()        
            
            rmse_list.append(rmse)
            median_l1_distance_list.append(median_l1_distance)
            cosine_similarity_list.append(cosine_similarity)
        
        else:
            [imputed_ari, imputed_nmi, imputed_ca], [reduced_ari, reduced_nmi, reduced_ca] = embedder.train()
        
        imputed_ari_list.append(imputed_ari)
        imputed_nmi_list.append(imputed_nmi)
        imputed_ca_list.append(imputed_ca)

        reduced_ari_list.append(reduced_ari)
        reduced_nmi_list.append(reduced_nmi)
        reduced_ca_list.append(reduced_ca)

    logger.info('')
    logger.info(datetime.datetime.now())
    logger.info(file)
    if args.drop_rate > 0.0:
        logger.info(f'-------------------- Drop Rate: {args.drop_rate} --------------------')
        logger.info('[Averaged result]  RMSE  Median_L1  Cosine_Similarity')
        logger.info('{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}'.format(np.mean(rmse_list), np.std(rmse_list), np.mean(median_l1_distance_list), np.std(median_l1_distance_list), np.mean(cosine_similarity_list), np.std(cosine_similarity_list)))    
    logger.info('[Averaged result] (Imputed) ARI  NMI  CA')
    logger.info('{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}'.format(np.mean(imputed_ari_list), np.std(imputed_ari_list), np.mean(imputed_nmi_list), np.std(imputed_nmi_list), np.mean(imputed_ca_list), np.std(imputed_ca_list)))
    logger.info('[Averaged result] (Reduced) ARI  NMI  CA')
    logger.info('{:.4f}+{:.4f} {:.4f}+{:.4f} {:.4f}+{:.4f}'.format(np.mean(reduced_ari_list), np.std(reduced_ari_list), np.mean(reduced_nmi_list), np.std(reduced_nmi_list), np.mean(reduced_ca_list), np.std(reduced_ca_list)))
    logger.info('')
    logger.info(args)
    logger.info(f'=================================')

if __name__ == "__main__":
    main()