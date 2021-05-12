# Ucuddle-Fast-PageRank


## Execution and testing

1. You can use our data or make sure that everything works - run pagerank_preview.py two times
   with different flag PERSONALIZATION_MODE = True and after PERSONALIZATION_MODE = True, 
   which is placed at the beginning of the file. Such you will get personalized_result_domain_ranks.json and result_domain_ranks.json, 
   which we use for testing correctness of personalization.
   
2. For testing correctness of ranks of our Fast Pagerank you can use our data or run it by yourself. 
   For this run src/get_methods_results.py to get comparison_domains_ranks.csv, which we will test soon.
   
3. Now it is the most interesting part -- testing. Run one by one cells in the Benchmarking_and_Testing.ipynb to check 
   correctness of the results and visualize data.
   
4. To look at our results on the real data, parsed from the Internet, execute PageRank_Real_Data_Values_Comparison.ipynb .


## Results

- results of the first version of algorithm (29.04.2021) you can find [here](https://github.com/denysgerasymuk799/Ucuddle-Fast-PageRank/blob/main/version1_result_domain_ranks.json)
  (json is sorted from highest to lowest ranks). Algorithm was tested only on 1000 web pages, but we have already noticed efficiency and
  right direction of ranks for special websites. 
  
- you can change PERSONALIZATION_MODE = True
