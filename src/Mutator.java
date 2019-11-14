import java.util.Random;

/**
 * Class implementing the Mutation operator for the GP-EMLC method.
 * Select a subtree (or leaf) and introduces a randomly created subtree (or leaf).
 * 
 * @author Jose M. Moyano
 *
 */
public class Mutator {
	
	/**
	 * Random numbers generator
	 */
	Random rand = new Random();
	
	/**
	 * Maximum value for leafs
	 */
	int nMax;
	
	/**
	 * Number of childs of each node
	 */
	int nChilds;
	
	/**
	 * Utils object for working with GP individuals
	 */
	Utils utils = new Utils();
	
	/**
	 * Maximum allowed depth for the tree
	 */
	int maxTreeDepth;
	
	/**
	 * Setter for the random number generator
	 * @param rand
	 */
	public void setRand(Random rand) {
		this.rand = rand;
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
	 * Setter for the number of children at each node
	 * @param nChilds
	 */
	public void setnChilds(int nChilds) {
		this.nChilds = nChilds;
	}

	/**
	 * Cross individuals and obtains two new sons.
	 * 
	 * @param ind1 First parent
	 * @param ind2 Second parent
	 * @return Array with two child individuals
	 */
	public String mutate(String ind) {
		boolean chooseLeaf = rand.nextBoolean();
		int[] subTree;
		
		if(chooseLeaf || utils.calculateTreeMaxDepth(ind) <= 1) {
			subTree = utils.chooseRandomLeaf(ind);
		}
		else {
			subTree = utils.chooseRandomSubTree(ind, false);
		}
		
		int allowedDepth = maxTreeDepth - utils.calculateNodeDepth(ind, subTree[0]);
		IndividualCreator creator = new IndividualCreator();
		String newSubtree = creator.create(nMax, allowedDepth, nChilds);
		
		//newSubtree.length()-1 is to remove the ";"
		return ind.substring(0, subTree[0]) + newSubtree.substring(0, newSubtree.length()-1) + ind.substring(subTree[1], ind.length());
		
		
	}
	
}
