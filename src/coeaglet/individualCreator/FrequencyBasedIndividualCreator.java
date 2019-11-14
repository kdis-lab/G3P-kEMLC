package coeaglet.individualCreator;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.DoubleStream;

import coeaglet.utils.Utils;
import net.sf.jclec.IIndividual;
import net.sf.jclec.listind.MultipListCreator;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;

/**
 * Class implementing the creation of individuals.
 * It is biased by the frequency of labels; the greater the frequency of a label the greater the appearances on the initial population.
 * Each label should appear at least a minimum number of times.
 * 
 * @author Jose M. Moyano
 *
 */
public class FrequencyBasedIndividualCreator extends MultipListCreator {

	/**
	 * Serialization constant
	 */
	private static final long serialVersionUID = -3389888931465909537L;
	
	/**
	 *  Array with the frequencies of each label in the dataset
	 */
	private int [] appearances;
	
	/**
	 * Minimum number of appearances per label
	 */
	private int aMin;
	
	/**
	 * Constructor
	 */
	public FrequencyBasedIndividualCreator()
	{
		super();
	}
	
	/**
	 * Sets the frequencies
	 * 
	 * @param frequencies Array with frequencies
	 */
	public void setAppearances(int [] appearances)
	{
		this.appearances = appearances;
	}
	
	/**
	 * Setter for aMin
	 * 
	 * @param aMin Minimum number of appearances of each label
	 */
	public void setaMin(int aMin) {
		this.aMin = aMin;
	}


	@Override
	public List<IIndividual> provide(int numberOfIndividuals) 
	{
		// Set numberOfIndividuals
		this.numberOfIndividuals = numberOfIndividuals;
		// Result list
		createdBuffer = new ArrayList<IIndividual> (numberOfIndividuals);
		// Prepare process
		prepareCreation();
		
		createdBuffer = generateIndividuals(numberOfIndividuals, subpop, appearances, aMin);
		
		// Returns result
		return createdBuffer;
	}
	
	/**
	 * Generate individuals for a given subpop
	 * 
	 * @param numberOfIndividuals Number of individuals
	 * @return List of generated IIndividuals
	 */
	public List<IIndividual> generateIndividuals(int numberOfIndividuals, int subpop, int [] appearances, int aMin){	
		//List of indiviuals
		List<IIndividual> inds = new ArrayList<IIndividual>(numberOfIndividuals);
		
		//Calculate initial weights given appearances
		double [] weights = new double[appearances.length];
		double sumAppearances = Arrays.stream(appearances).sum();		
		double [] baseVotes = new double[appearances.length];
		
		for(int i=0; i<weights.length; i++) {
			//Expected votes per label is the ratio of appearance of the label divided by the number of bits to share
			baseVotes[i] = (appearances[i] / sumAppearances) * (numberOfIndividuals*k);
			
			//If the expected votes is lower than the minimum, set the minimum
			if(baseVotes[i] < aMin) {
				weights[i] = aMin;
			}
			//The expected votes are upper bounded by the number of individuals
			else if(baseVotes[i] > numberOfIndividuals) {
				weights[i] = numberOfIndividuals;
			}
			else {
				weights[i] = baseVotes[i];
			}
		}
		
		//Calculate probabilities given weights
		double [] prob = new double[appearances.length];
		double sumWeights = DoubleStream.of(weights).sum();
		for(int i=0; i<weights.length; i++) {
			prob[i] = weights[i] / sumWeights;
		}
		
		//Create individuals
		ArrayList<Integer> list = new ArrayList<Integer>(k);
		int index;
		double [] probCopy;
		
		//Until the desired number of individuals is created
		while(inds.size() < numberOfIndividuals) {
			//Clear list of genotype
			list.clear();
			
			//Copy array of probabilities
			probCopy = prob.clone();

			while(list.size() < k) {
				//Select random index (based on probs) to add to the individual
				index = Utils.probabilitySelectIndex(probCopy, randgen);
				
				//Add index to the individual
				list.add(index);
				
				//Give 0 probability to the same index to be selected
				probCopy[index] = 0;
			}
			
			//Sort individual
			Collections.sort(list);
			
			//Add individual to the list if it still does not exist
			MultipListIndividual newInd = new MultipListIndividual(new MultipListGenotype(subpop, new ArrayList<Integer>(list)));
			if(!Utils.contains(inds, newInd))
			{	
				inds.add(newInd);
			}
		}
		
		//Check if all labels appear at least aMin times
		int [] labelAppearances = getAppearancesLabels(inds);
		
		for(int i=0; i<labelAppearances.length; i++) {
			//If a label appear less than aMin
			while(labelAppearances[i] < aMin) {
				//Get an individual including the label that most appear in the population
				IIndividual maxLabelInd = getMaxLabelIndividual(inds, labelAppearances, aMin);
				int p = ((MultipListIndividual)maxLabelInd).getGenotype().subpop;
				ArrayList<Integer> gen = ((MultipListIndividual)maxLabelInd).getGenotype().genotype;
				
				//Get index of max label
				int maxIndex = 0;
				for(int j=1; j<gen.size(); j++) {
					if(labelAppearances[gen.get(j)] > labelAppearances[maxIndex]) {
						maxIndex = j;
					}
				}
				
				//Quit index of max label and add i-th label to genotype
				labelAppearances[i]++;
				labelAppearances[gen.get(maxIndex)]--;
				gen.remove(maxIndex);
				gen.add(i);
				Collections.sort(gen);
				
				//Remove old individual and add new one to subpopulation
				inds.remove(maxLabelInd);
				inds.add(new MultipListIndividual(new MultipListGenotype(p, gen)));
			}
		}
		
		System.out.println("labelAppearances: " + Arrays.toString(labelAppearances));
		
		return inds;
	}
	
	/**
	 * Get the appearances of each label in the population
	 * 
	 * @param inds Population of individuals
	 * @return Appearances of each label in this population
	 */
	protected int [] getAppearancesLabels(List<IIndividual> inds) {
		int [] appearances = new int[maxInt];
		
		for(IIndividual ind : inds) {
			for(int g : ((MultipListIndividual)ind).getGenotype().genotype) {
				appearances[g]++;
			}
		}
		
		return appearances;
	}
	
	/**
	 * Get an individual with maximum label (label that most appear), but not critical label (labels appearing <= aMin)
	 * 
	 * @param inds List of individuals
	 * @param appearances Appearances of each label in the population
	 * @param aMin Minimum appearances for each label
	 * @return Individual with max label
	 */
	protected IIndividual getMaxLabelIndividual(List<IIndividual> inds, int[] appearances, int aMin) {
		ArrayList<Integer> maxLabels = new ArrayList<Integer>();
		ArrayList<Integer> criticalLabels = new ArrayList<Integer>();
		
		maxLabels.add(0);
		for(int i=1; i<appearances.length; i++) {
			if(appearances[i] > appearances[maxLabels.get(0)]) {
				maxLabels.clear();
				maxLabels.add(i);
			}
			else if(appearances[i] == appearances[maxLabels.get(0)]) {
				maxLabels.add(i);
			}
			else if(appearances[i] <= aMin) {
				criticalLabels.add(i);
			}
		}
		
		int maxLabel = maxLabels.get(randgen.choose(0, maxLabels.size()));
		
		List<IIndividual> indsMaxLabel = new ArrayList<IIndividual>();
		//Add individual with max label and not critical label
		for(IIndividual ind : inds) {
			if(((MultipListIndividual)ind).getGenotype().genotype.contains(maxLabel) && !containsCriticalLabel((MultipListIndividual)ind, criticalLabels)) {
				indsMaxLabel.add(ind);
			}
		}
		
		//Return one of these individuals
		return indsMaxLabel.get(randgen.choose(0, indsMaxLabel.size()));
	}
	
	/**
	 * Check if a given individual contains a critical label
	 * 
	 * @param ind Individual
	 * @param criticalLabels Set of critical labels
	 * @return true if ind contains any of the critical labels and false otherwise
	 */
	protected boolean containsCriticalLabel(MultipListIndividual ind, List<Integer> criticalLabels) {
		for(Integer g : ind.getGenotype().genotype) {
			if(criticalLabels.contains(g)) {
				return true;
			}
		}
		
		return false;
	}
	
}
