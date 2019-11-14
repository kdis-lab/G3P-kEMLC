package gpemlc;

import java.util.Random;

/**
 * Class implementing the Crossover operator for the GP-EMLC method.
 * Swaps sub-trees among individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class Crossover {
	
	/**
	 * Random numbers generator
	 */
	Random rand = new Random();
	
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
	 * Cross individuals and obtains two new sons.
	 * 
	 * @param ind1 First parent
	 * @param ind2 Second parent
	 * @return Array with two child individuals
	 */
	public String[] cross(String ind1, String ind2) {
		String[] crossedInds = new String[2];

		crossedInds[0] = crossInd(ind1, ind2);
		crossedInds[1] = crossInd(ind2, ind1);
		return crossedInds;
	}
	
	/**
	 * Cross two individuals to obtain only one.
	 * Select a random subtree (or leaf) in first parent
	 * 	and swap by another subtree (that fits) of the second parent
	 * 
	 * @param ind1 First parent
	 * @param ind2 Second parent
	 * @return Child individual
	 */
	private String crossInd(String ind1, String ind2) {
		boolean chooseLeaf = rand.nextBoolean();
		int[] subTree = new int[2];
		int[] otherSubtree = new int[2];
		int allowedDepth, currentDepth;
		String introduceSubtree;

		//Choose leaf or subtree of parent
		if(chooseLeaf || utils.calculateTreeMaxDepth(ind1) <= 1) { //If the parent has depth=1, we can just choose leaf
			subTree = utils.chooseRandomLeaf(ind1);
//			System.out.println("leaf1: " + ind1.substring(subTree[0], subTree[1]));
		}
		else {
			subTree = utils.chooseRandomSubTree(ind1, false);
//			System.out.println("subtree1: " + ind1.substring(subTree[0], subTree[1]));
		}
		
		//Calculate max depth allowed for subtree to select from the other ind
		allowedDepth = maxTreeDepth - utils.calculateNodeDepth(ind1, subTree[0]);
		if(allowedDepth == 0) { //We can just select a leaf
			otherSubtree = utils.chooseRandomLeaf(ind2);
			introduceSubtree = ind2.substring(otherSubtree[0], otherSubtree[1]);
//			System.out.println("leaf2: " + introduceSubtree);
		}
		else {
			chooseLeaf = rand.nextBoolean();
			if(chooseLeaf) { //Choose leaf
				otherSubtree = utils.chooseRandomLeaf(ind2);
				introduceSubtree = ind2.substring(otherSubtree[0], otherSubtree[1]);
//				System.out.println("leaf2: " + introduceSubtree);
			}
			else { //Choose subtree until allowed size
				String cpInd2 = new String(ind2);
				otherSubtree[0] = 0;
				otherSubtree[1] = cpInd2.length();
				do {
//					System.out.println("cpInd2: " + cpInd2);
					cpInd2 = cpInd2.substring(otherSubtree[0], otherSubtree[1]);
					otherSubtree = utils.chooseRandomSubTree(cpInd2, true);
					currentDepth = utils.calculateTreeMaxDepth(cpInd2.substring(otherSubtree[0], otherSubtree[1]));
//					System.out.println(currentDepth + " ; " + allowedDepth);
				}while(currentDepth > allowedDepth);
				introduceSubtree = cpInd2.substring(otherSubtree[0], otherSubtree[1]);
//				System.out.println("subtree2: " + introduceSubtree);
			}
		}
		
//		return ind1.substring(0, subTree[0]) + ind2.substring(otherSubtree[0], otherSubtree[1]) + ind1.substring(subTree[1], ind1.length());
		return ind1.substring(0, subTree[0]) + introduceSubtree + ind1.substring(subTree[1], ind1.length());
	}
}
