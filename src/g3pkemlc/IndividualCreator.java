package g3pkemlc;

import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the individual creator for G3P-kEMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class IndividualCreator {

	/**
	 * Random numbers generator
	 */
	IRandGen randgen;
	
	/**
	 * Constructor
	 * 
	 * @param randgen Random numbers generator
	 */
	public IndividualCreator(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Setter for randgen
	 * @param rangen
	 */
	public void setRangen(IRandGen randgen) {
		this.randgen = randgen;
	}
	
	/**
	 * Create a tree individual of given maximum depth and maximum number of children in each node
	 * 
	 * @param nMax Max value for the leaves
	 * @param maxDepth Maximum depth of the tree
	 * @param maxChild Maximum number of children at each combination node
	 * @return Individual as String
	 */
	public String create(int nMax, int maxDepth, int maxChild) {
		//The individual at the beginning is the node 'S' and the end of individual ';'
		String ind = "S;";
		
		//Current position creating the individual
		int pos = 0;
		
		//Depth of the current node
		int currDepth = 0;
				
		do {
			switch (ind.charAt(pos)) {
			//If initial 'S' node, create a random number of child nodes 'C'.
				//Random in [2, maxChild] range
			case 'S':
				ind = replace(ind, pos, childRandomSize(maxChild));
				break;
			
			//If a node 'C' is found, it is replaced by one of:
			//	- Random number of child in [2, maxChild] range
			//	- Random leaf
			//If the current depth is equal than the max allowed depth, only a random leaf is allowed
			case 'C':
				if(currDepth < maxDepth) {
					if(randgen.coin()) {
						ind = replace(ind, pos, childRandomSize(maxChild));
					}
					else {
						//Replace current 'C' by random leaf in [0, nMax) range
						ind = replace(ind, pos, String.valueOf(randgen.choose(0, nMax)));
					}
				}
				else {
					//Replace current 'C' by random leaf in [0, nMax) range
					ind = replace(ind, pos, String.valueOf(randgen.choose(0, nMax)));
				}				
				break;
				
			//If a start parenthesis is found, a new node starts so the current depth is incremented
			case '(':
				currDepth++;
				pos++;
				break;
			
			//If a close parenthesis is found, a node ends so the current depth is decremented
			case ')':
				currDepth--;
				pos++;
				break;
				
			//If any other character, just go to next position in the String
			default:
				pos++;
				break;
			}
		}while(ind.charAt(pos) != ';'); //Finish when the end character ";" is found
		
		return ind;
	}
	
	/**
	 * Creates a child in the form (C ... C) including n times the 'C'
	 * 
	 * @param n Number of children
	 * @return String
	 */
	private String child(int n) {
		//Start with open parenthesis
		String child = "(";
		
		//Add n times the "C" and a space
		for(int i=0; i<n; i++) {
			child += "C ";
		}
		
		//Replace last space " " with end parenthesis
		child = child.substring(0, child.length()-1) + ")";
		
		return child;
	}
	
	/**
	 * Creates a child in the form (C ... C) including the 'C' a random number of times between [2, max]
	 * 
	 * @param maxChildren Max number of children
	 * @return String
	 */
	private String childRandomSize(int maxChildren) {
		return child(randgen.choose(2, maxChildren+1)); //between 2 and max, included
	}
	
	/**
	 * Replace a given position in the string with a new string
	 * 
	 * @param str Original string 
	 * @param pos Position to replace
	 * @param newStr String to insert in position 'pos'
	 * @return Modified string
	 */
	public String replace(String str, int pos, String newStr) {
		return str.substring(0, pos) + newStr + str.substring(pos+1, str.length());
	}
}
