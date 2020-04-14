package g3pkemlc.mutator;

import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import g3pkemlc.IndividualCreator;
import g3pkemlc.utils.Utils;
import net.sf.jclec.ISpecies;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.stringtree.StringTreeMutator;
import net.sf.jclec.stringtree.StringTreeSpecies;

/**
 * Class implementing the double mutator.
 * It mutates either a (number of) threshold(s), or a subtree
 * 
 * @author Jose M. Moyano
 *
 */
public class DoubleMutator extends StringTreeMutator {

	/** Serialization constant */
	private static final long serialVersionUID = 2455293830055566959L;
	
	/** Individual species (taken from execution context) */
	protected transient StringTreeSpecies species;
	
	/** Genotype schema */ 	
	protected transient String schema;

	/**
	 * Ratio of tresholds of the tree to mutate
	 */
	protected double ratioTresholdsToMutate;
	
	/**
	 * Maximum value for leafs
	 */
	int nMax;
	
	/**
	 * Min number of children of each node
	 */
	int minChildren;
	
	/**
	 * Max number of children of each node
	 */
	int maxChildren;
	
	/**
	 * Utils object for working with GP individuals
	 */
	Utils utils = new Utils();
	
	/**
	 * Maximum allowed depth for the tree
	 */
	int maxTreeDepth;
	
	/**
	 * Standard deviation parameter for gaussian to calculate thresholds
	 */
	double stdvGaussianThreshold;
	
	/**
	 * Constructor
	 */
	public DoubleMutator()
	{
		super();
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof DoubleMutator) {
			return true;
		}
		else {
			return false;
		}
	}

	/**
	 * Set the ratio of thresholds to mutate in the individual
	 * 
	 * @param ratioTresholdsToMutate Ratio of thresholds to mutate
	 */
	public void setRatioTresholdsToMutate(double ratioTresholdsToMutate) {
		this.ratioTresholdsToMutate = ratioTresholdsToMutate;
	}
	
	/**
	 * Setter for the maximum tree depth
	 * @param maxTreeDepth
	 */
	public void setMaxTreeDepth(int maxTreeDepth) {
		this.maxTreeDepth = maxTreeDepth;
	}
	
	/**
	 * Setter for the maximum value of leafs
	 * @param nMax
	 */
	public void setnMax(int nMax) {
		this.nMax = nMax;
	}
	
	/**
	 * Set the minimum number of children at each node
	 * @param minChildren Min number of children at each node
	 */
	public void setMinChildren(int minChildren) {
		this.minChildren = minChildren;
	}
	
	/**
	 * Set the maximum number of children at each node
	 * @param maxChildren Max number of children at each node
	 */
	public void setMaxChildren(int maxChildren) {
		this.maxChildren = maxChildren;
	}
	
	/**
	 * Set the standard deviation for the gaussian method to calculate thresholds
	 * @param stdvGaussianThreshold Standard deviation for gaussian
	 */
	public void setStdvGaussianThreshold(double stdvGaussianThreshold) {
		this.stdvGaussianThreshold = stdvGaussianThreshold;
	}

	/**
	 * Mutate next individual 
	 */
	protected void mutateNext() {
		//Get individual to be mutated
		StringTreeIndividual mutant = (StringTreeIndividual) parentsBuffer.get(parentsCounter);
		
		//Mutate individual and add to buffer
		sonsBuffer.add(new StringTreeIndividual(mutate(mutant.getGenotype())));
	}
	
	/**
	 * Mutate an individual.
	 * It mutates either one/several thresholds or a subtree
	 * 
	 * @param ind Individual to mutate
	 * @return Mutated individual
	 */
	public String mutate(String ind) {
		if(randgen.coin()) {
			return thresholdMutator(ind);
		}
		else {
			return subtreeMutator(ind);
		}
	}
	
	/**
	 * Mutate one or several thresholds of an individual
	 * 
	 * @param ind Individual to mutate
	 * @return Mutated individual
	 */
	protected String thresholdMutator(String ind) {
		//Number of thresholds in the individual
		int nThresholds = utils.countParenthesis(ind);
				
		//Number of thresholds to mutate
		int nTresholdsMutate = (int) Math.round(nThresholds * ratioTresholdsToMutate);
		if(nTresholdsMutate < 1) {
			nTresholdsMutate = 1;
		}
				
		int randomThres;
		int count;
		double newThreshold;
		Pattern p = Pattern.compile("v\\.\\d+ "); // "\d" is for digits in regex
		Matcher m;
				
		//Mutate nTresholdsMutate times
		for(int i=0; i<nTresholdsMutate; i++) {
			//Select a random threshold
			randomThres = randgen.choose(0, nThresholds);
			
			//Match thresholds until reached the desired one
			m = p.matcher(ind);
			count = 0;
			while(count <= randomThres){
				m.find();
				count++;
			}
					
			//Set new threshold
			newThreshold = utils.randomThreshold(stdvGaussianThreshold);
			ind = ind.substring(0, m.start()) + "v." + String.format(Locale.US, "%.2f", newThreshold).split("\\.")[1] + " " + ind.substring(m.end(), ind.length());
		}
		
		return ind;
	}
	
	/**
	 * Mutate an individual by removing a random subtree by a new one randomly created
	 * 
	 * @param ind Individual to mutate
	 * @return Mutated individual
	 */
	String subtreeMutator(String ind) {
		//Selected subtree
		int[] subTree;
		
		//Choose a leaf if the coin said that, or if the depth of the tree is 1 (just leaves)
		if(randgen.coin() || utils.calculateTreeMaxDepth(ind) <= 1) {
			subTree = utils.chooseRandomLeaf(ind);
		}
		else {
			subTree = utils.chooseRandomSubTree(ind, false);
		}
		
		//Calculate the allowed depth for the subtree to include in the current position
		int allowedDepth = maxTreeDepth - utils.calculateNodeDepth(ind, subTree[0]);
		String newSubtree;
		
		//If allowedDepth is 0, or if it was determined by the coin, replace by a leaf (select just a random leaf)
		if(allowedDepth == 0 || randgen.coin()) {
			newSubtree = String.valueOf(randgen.choose(0, nMax));
		}
		else {
			//Create a new subtree of maximum allowedDepth to substitute the node
			IndividualCreator creator = new IndividualCreator(randgen);
			newSubtree = creator.create(nMax, allowedDepth, minChildren, maxChildren, stdvGaussianThreshold);
			
			//Remove here the ";" indicating the end of the individual
			newSubtree = newSubtree.substring(0, newSubtree.length()-1);
		}
		
		//Create new individual by combining old and new subtree
		return ind.substring(0, subTree[0]) + newSubtree + ind.substring(subTree[1], ind.length());
	}
	
	@Override
	protected void prepareMutation() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof StringTreeSpecies) {
			// Set individuals species
			this.species = (StringTreeSpecies) species;
			// Sets genotype schema
			this.schema = this.species.getGenotypeSchema();
		}
		else {
			throw new IllegalStateException("Invalid species in context");
		}
	}
	
}
