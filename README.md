Trajectory Prediction of Traffic Agents at Urban Intersections Through Learned Interactions
-------------


Companion site for the ITSC 2017 paper : *Trajectory Prediction of Traffic Agents at Urban Intersections Through Learned Interactions*

----------


FAQs
-------------

 1\.  **Where are the videos?**

	Under `videos/`
	


----------
2\.  **Where are the annotations?**

	`trajectories\.db` is a sqlite3 database that contains the annotation for the trajectories. You might want to check out `trajectories-schema\.svg` to get an idea of the relevant tables and columns.
	


----------
3\.  **Where is the code?**

Influence net is under `NINnfluenceEng`, and the Dynamic Bayesian Network is under `DBNprediction`


----------
4\. **How to run the code to see the results of the papers**

Run `read_n_show\.py` under NINnfluenceEng directory. The script reads from the saved results to show and compare the models.

P\.S: If you want to run the prediction model, you would need to execute `predict\.py` (under DBNprediction) for the Dynamic Bayesian Network, and `nn-predict\.py` (under NINnfluenceEng) for the Influence Network.

	
