package coeaglet.algorithm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import coeaglet.utils.Utils;
import net.sf.jclec.IIndividual;
import net.sf.jclec.fitness.SimpleValueFitness;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the selection of members for an ensemble given a population of individuals.
 * The size of the population should be greater or equal than the desired ensemble size.
 * 
 * @author Jose M. Moyano
 *
 */
public class EnsembleSelection {

	/**
	 * Population of individuals to be selected
	 */
	List<IIndividual> population;
	
	/**
	 * Selected ensemble
	 */
	List<IIndividual> ensemble;
	
	/**
	 * Number of labels
	 */
	int nLabels;
	
	/**
	 * Size of the current ensemble
	 */
	int currentEnsembleSize;
	
	/**
	 * Desired ensemble size
	 */
	int desiredEnsembleSize;
	
	/**
	 * Current votes for each label
	 */
	int [] labelVotes;

	/**
	 * Beta value to weight the diversity in the ensemble selection
	 */
	double beta;
	
	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Constructor
	 * 
	 * @param population Population of possible members
	 * @param desiredEnsembleSize Size of the desired ensemble
	 * @param nLabels Number of labels of the dataset
	 * @param beta Beta value to give more importance to the diversity or to the fitness
	 */
	public EnsembleSelection(List<IIndividual> population, int desiredEnsembleSize, int nLabels, double beta) {
		//Copy all individuals
		this.population = new ArrayList<IIndividual>(population.size());
		for(int i=0; i<population.size(); i++) {
			this.population.add(population.get(i).copy());
		}
		
		this.desiredEnsembleSize = desiredEnsembleSize;
		this.nLabels = nLabels;
		this.beta = beta;
		
		this.ensemble = new ArrayList<IIndividual>(desiredEnsembleSize);
		this.labelVotes = new int[nLabels];
	}
	
	/**
	 * Setter for randgen; Random number generator
	 * 
	 * @param randgen Randgen
	 */
	public void setRandgen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Getter for the ensemble
	 * 
	 * @return Ensemble
	 */
	public List<IIndividual> getEnsemble() {
		return ensemble;
	}
	
	/**
	 * Getter for the current ensemble size
	 * 
	 * @return Current ensemble size
	 */
	public int getCurrentEnsembleSize() {
		return currentEnsembleSize;
	}
	
	/**
	 * Getter for the current number of votes per label
	 * 
	 * @return Array with votes per label
	 */
	public int[] getLabelVotes() {
		return labelVotes;
	}
	
	/**
	 * Selects members for the ensemble previously given the population and desired size
	 */
	public void selectEnsemble() {
		//Initialize
		this.ensemble = new ArrayList<IIndividual>(desiredEnsembleSize);
		this.labelVotes = new int[nLabels];

		//Get best individual
		int best = -1;
		double bestFit = -1;
		IIndividual currInd = null;
		for(int i=0; i<population.size(); i++) {
			currInd = population.get(i);
			
			if(((SimpleValueFitness)currInd.getFitness()).getValue() > bestFit) {
				bestFit = ((SimpleValueFitness)currInd.getFitness()).getValue();
				best = i;
			}
		}
		
		//Add best to ensemble
		ensemble.add(population.get(best));
		
		//Update votes
		for(Integer g : ((MultipListIndividual)population.get(best)).getGenotype().genotype) {
			labelVotes[g]++;
		}
		
		//Remove best from population
		population.remove(best);
		
		//Until desired ensemble size is reached, select individuals based on fitness and diversity
		while(ensemble.size() < desiredEnsembleSize) {
			//Re-calculated fitnesses
			double [] newFitness = new double[population.size()];

			//Weights for distance
			double [] weights = new double[nLabels];
			for(int l=0; l<nLabels; l++) {
				if(labelVotes[l] < 1) {
					weights[l] = 1.0 / 0.5; //If it still does not appear, probability is double than if appear once
				}
				else {
					weights[l] = 1.0 / labelVotes[l];
				}
			}
			double sumW = Arrays.stream(weights).sum();
			for(int l=0; l<nLabels; l++) {
				weights[l] /= sumW;
			}
						
			//Calculate new fitness including distance of each member
			double dist = -1;
			for(int i=0; i<population.size(); i++) {
				dist = 0;
				for(Integer g : ((MultipListIndividual)population.get(i)).getGenotype().genotype) {
					dist += weights[g];
				}
				
				//New fitness is: B*dist + (1-B)*fitness
				newFitness[i] = (beta * dist) + ((1-beta) * ((SimpleValueFitness)population.get(i).getFitness()).getValue());
			}
			
			//Get index of individual with max newFitness
			best = Utils.getMaxIndex(newFitness, randgen);
			
			//Add current best to ensemble
			ensemble.add(population.get(best));
			
			//Update votes
			for(Integer g : ((MultipListIndividual)population.get(best)).getGenotype().genotype) {
				labelVotes[g]++;
			}
			
			//Remove current best from population
			population.remove(best);
		}
	}
	
}
