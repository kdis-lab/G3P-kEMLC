package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Hashtable;
import java.util.List;

import mulan.classifier.InvalidDataException;
import mulan.classifier.ModelInitializationException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.classifier.transformation.LabelPowerset2;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.IIndividual;
import net.sf.jclec.listind.MultipListIndividual;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.TechnicalInformation;

/**
 * Class implementing the multi-label ensemble.
 * It is based on the combination of multi-label classifiers each described by a MultipListIndividual
 * 
 * @author Jose M. Moyano
 *
 */
public class Ensemble extends MultiLabelMetaLearner {

	/**
	 * Serial ID
	 */
	private static final long serialVersionUID = -5489297040283594557L;

	/**
	 * Ensemble
	 */
	protected MultiLabelLearner [] ensemble;
	
	/**
	 * Individuals that compose the ensemble
	 */
	protected ArrayList<MultipListIndividual> inds;
	
	/**
	 * Number of classifiers in the ensemble
	 */
	public int nClassifiers;
	
	/**
	 * Classifiers built so far
	 */
	Hashtable<String, MultiLabelLearner> tableClassifiers;
	
	/**
	 * Fitness of the ensemble
	 */
	double fitness;

	
	/**
	 * Constructor 
	 * 
	 * @param baseLearner Multi-label base learner
	 */
	public Ensemble(MultiLabelLearner baseLearner) {
		super(baseLearner);
		
		fitness = -1;
	}

	/**
	 * Constructor with individuals and learner.
	 * 
	 * @param inds Individuals to form the ensemble
	 * @param baseLearner Multi-label base learner
	 */
	public Ensemble(List<IIndividual> inds, MultiLabelLearner baseLearner) {
		super(baseLearner);
		
		//Copy individuals
		this.inds = new ArrayList<MultipListIndividual>(inds.size());
		for(IIndividual ind : inds) {
			this.inds.add((MultipListIndividual)ind.copy());
		}
		
		nClassifiers = inds.size();
		
		fitness = -1;
	}

	/**
	 * Setter for tableClassifiers
	 * 
	 * @param tableClassifiers Table storing all classifiers built so far
	 */
	public void setTableClassifiers(Hashtable<String, MultiLabelLearner> tableClassifiers) {
		this.tableClassifiers = tableClassifiers;
	}
	
	/**
	 * Reset seed for each member
	 */
	public void resetSeed() {
		for(int i=0; i<nClassifiers; i++) {
			((LabelPowerset2)ensemble[i]).setSeed(1);
		}
	}
	
	/**
	 * Setter for fitness
	 * 
	 * @param fitness Fitness value computed by EnsembleEval
	 */
	public void setFitness(double fitness) {
		this.fitness = fitness;
	}
	
	@Override
	/**
	 * Creates an array of MultiLabelLearners, i.e., the ensemble.
	 * It obtains previously created classifiers from the table.
	 */
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {		
		//The number of classifier is determined by the number of individuals
		nClassifiers = inds.size();
		
		//Ensemble members should be already in the table
		ensemble = new MultiLabelLearner[nClassifiers];
		for(int i=0; i<nClassifiers; i++) {
			ensemble[i] = tableClassifiers.get(((MultipListIndividual)inds.get(i)).getGenotype().toString());
		}
		
	}


	/**
	 * Make prediction for a given instance.
	 * 
	 * It combines all available votes for each label with majority voting.
	 */
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
		
		double [] sumVotes = new double[numLabels];
		double [] sumConf = new double[numLabels];
		int [] nVotes = new int[numLabels];
		MultiLabelOutput subsetMLO;
		
		int index;
		for(int i=0; i<nClassifiers; i++) {
			//Predict
			subsetMLO = ensemble[i].makePrediction(instance);
			index = 0;
			for(int j : inds.get(i).getGenotype().genotype) {
				sumVotes[j] += subsetMLO.getBipartition()[index] ? 1 : 0;
				sumConf[j] += subsetMLO.getConfidences()[index];
				nVotes[j]++;
				index ++;
			}
		}
		
		//Not ever more used
		subsetMLO = null;
		
		boolean[] bipartition = new boolean[numLabels];
		double[] conf = new double[numLabels];
		double threshold = 0.5;
		
		for(int i=0; i<numLabels; i++) {
			if(((sumVotes[i]*1.0) / nVotes[i]) >= threshold) {
				bipartition[i] = true;
			}
			else {
				bipartition[i] = false;
			}
			
			conf[i] = sumConf[i] / nVotes[i];
		}

		MultiLabelOutput mlo = new MultiLabelOutput(bipartition, conf);
		
		return mlo;
	}
	
	/**
	 * Tries to prune an ensemble.
	 * It tries to prune each member (starting from the last) and remove it only if fitness is better
	 * 
	 * @return Number of pruned members
	 */
	public int prune(MultiLabelInstances mlData) {
		//Number of pruned members
		int nPruned = 0;
		
		//Current ensemble should have been already evaluated
		double bestFitness = fitness;
		
		//Evaluator
		EnsembleEval eEval = new EnsembleEval(this, mlData);
		double currFitness;
		
		//Individual to remove at each iteration
		MultipListIndividual toRemove;
		
		try {
			//Try to prune each member
	     	for(int i=(inds.size()-1); i>=0; i--) {
	     		//Remove corresponding member and store to maybe include it again
	     		toRemove = inds.get(i);
	     		inds.remove(i);
	     		
	     		//Build and evaluate ensemble with current members
				this.build(mlData);
				eEval = new EnsembleEval(this, mlData);
				currFitness = eEval.evaluate();
	     		
		     	//If improves, set the new fitness; if not, add the member in the same position
		     	if(currFitness > bestFitness) {
		     		bestFitness = currFitness;
		     		nPruned++;
		     	}
		     	else {
		     		inds.add(i, toRemove);
		     	}
	     	}
	     	
	     	//Object no longer used
	     	eEval = null;
	     	toRemove = null;
	     	
	     	//Finally, re-build with final members
	     	this.build(mlData);
	     	
	     	//Fitness of the current ensemble should be bestFitness
	     	fitness = bestFitness;
		}
		catch(Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
		
		//Return the number of pruned members
		return nPruned;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		return null;
	}

	@Override
	public String globalInfo() {
		return null;
	}
	
	/**
	 * Classify a multi-label dataset
	 * 
	 * @param mlData Multi-label dataset
	 * @return Matrix with the label predictions for all instances
	 */
	public int[][] classify(MultiLabelInstances mlData)
	{
		this.resetSeed();
		int[][] predictions = new int[mlData.getNumInstances()][numLabels];

		Instances data = mlData.getDataSet();
		
		for (int i=0; i<mlData.getNumInstances(); i++)
		{ 	
		    try {
				MultiLabelOutput mlo = this.makePrediction(data.get(i));
				for(int j=0; j<this.numLabels; j++)
				{	
				  if(mlo.getBipartition()[j])
				  {
					  predictions[i][j]=1;
				  }	
				  else
				  {
					  predictions[i][j]=0;
				  }	  
				}
				
			} catch (InvalidDataException e) {
				e.printStackTrace();
			} catch (ModelInitializationException e) {
				e.printStackTrace();
			} catch (Exception e) {
				e.printStackTrace();
			}
		    
		}
		
		return(predictions);		
	}	
	
	/**
	 * Get the votes for each label in the ensemble
	 * 
	 * @return Array with number of votes for each label
	 */
	public int[] getLabelVotes() {
		int [] labelVotes = new int[numLabels];
		
		for(MultipListIndividual ind : inds) {
			for(int g : ind.getGenotype().genotype) {
				labelVotes[g]++;
			}
		}
		
		return labelVotes;
	}
	
	@Override
	public String toString() {
		String s = new String();
		
		s += nClassifiers + " classifiers.\n";
		s += "Votes: " + Arrays.toString(getLabelVotes()) + "\n";
		s += "Ensemble: \n";
		for(MultipListIndividual ind : inds) {
			s += "\t" + ind.toString() + "\n";
		}
		
		return s;
	}

}
