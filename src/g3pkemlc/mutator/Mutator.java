package g3pkemlc.mutator;

import g3pkemlc.IndividualCreator;
import g3pkemlc.utils.Utils;
import net.sf.jclec.ISpecies;
import net.sf.jclec.stringtree.StringTreeIndividual;
import net.sf.jclec.stringtree.StringTreeMutator;
import net.sf.jclec.stringtree.StringTreeSpecies;

/**
 * Class implementing the mutator
 * 
 * @author Jose M. Moyano
 *
 */
public class Mutator extends StringTreeMutator {

	/** Serialization constant */
	private static final long serialVersionUID = 2455293830055566959L;
	
	/** Individual species (taked from execution context) */
	protected transient StringTreeSpecies species;
	
	/** Genotype schema */ 	
	protected transient String schema;

	/**
	 * Maximum value for leafs (number of classifiers in the pool)
	 */
	int nMax;
	
	/**
	 * Number of children of each node
	 */
	int maxChildren;
	
	/**
	 * Utils object for working with GP individuals
	 */
	Utils utils = new Utils();
	
	/**
	 * Maximum allowed depth for the tree
	 */
	int maxDepth;
	
	
	/**
	 * Constructor
	 */
	public Mutator()
	{
		super();
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof Mutator) {
			return true;
		}
		else {
			return false;
		}
	}
	

	/**
	 * Setter for the maximum tree depth
	 * @param maxTreeDepth Maximum tree depth
	 */
	public void setMaxTreeDepth(int maxTreeDepth) {
		this.maxDepth = maxTreeDepth;
	}
	
	/**
	 * Setter for the maximum value of leafs (number of classifiers in the pool)
	 * @param nMax Maximum value for leafs
	 */
	public void setnMax(int nMax) {
		this.nMax = nMax;
	}
	
	/**
	 * Setter for the number of children at each node
	 * @param maxChildren Max number of children at each node
	 */
	public void setnChildren(int maxChildren) {
		this.maxChildren = maxChildren;
	}

	/**
	 * Mutate next individual
	 * 
	 * This mutator randomly changes one value in the list for another random value
	 */
	protected void mutateNext() {
		//Get individual to be mutated
		StringTreeIndividual mutant = (StringTreeIndividual) parentsBuffer.get(parentsCounter);
		
		//Mutate individual and add to buffer
		sonsBuffer.add(new StringTreeIndividual(mutate(mutant.getGenotype())));
	}
	
	/**
	 * Mutate individual
	 * 
	 * @param ind Individual to mutate
	 * @return Mutated individual
	 */
	public String mutate(String ind) {
		boolean chooseLeaf = randgen.coin();
		int[] subTree;
		
		//Choose a leaf if the coin said that, or if the depth of the tree is 1 (just leaves)
		if(chooseLeaf || utils.calculateTreeMaxDepth(ind) <= 1) {
			subTree = utils.chooseRandomLeaf(ind);
		}
		else {
			subTree = utils.chooseRandomSubTree(ind, false);
		}
		
		//Calculate the allowed depth for the subtree to include in the current position
		int allowedDepth = maxDepth - utils.calculateNodeDepth(ind, subTree[0]);
		String newSubtree;
		
		boolean replaceByLeaf = randgen.coin();
		//If allowedDepth is 0, or if it was determined that the replacement is a leaf, select just a random leaf
		if(allowedDepth == 0 || replaceByLeaf) {
			newSubtree = String.valueOf(randgen.choose(0, nMax));
		}
		else {
			//Create a new subtree of maximum allowedDepth to substitute the node
			IndividualCreator creator = new IndividualCreator(randgen);
			newSubtree = creator.create(nMax, allowedDepth, maxChildren);
			
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
