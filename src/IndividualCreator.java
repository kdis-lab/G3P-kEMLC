import java.util.Random;

/**
 * Class implementing the individual creator for GP-EMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class IndividualCreator {
	
	/**
	 * Random numbers generator
	 */
	static Random rand = new Random();

	
	/**
	 * Create an individual of given maximum depth
	 * 
	 * @param nMax Max value for the leaves
	 * @param maxDepth Maximum depth of the tree
	 * @return Individual as String
	 */
	public String create(int nMax, int maxDepth, int nChilds) {
		String ind = "S;";
		int pos = 0;
		int currDepth = 0;
				
		do {
			switch (ind.charAt(pos)) {
			case 'S':
				ind = replace(ind, pos, childRandomSize(nChilds));
				break;
			case 'C':
				if(currDepth < maxDepth) {
					if(rand.nextBoolean()) {
						ind = replace(ind, pos, childRandomSize(nChilds));
					}
					else {
						ind = replace(ind, pos, String.valueOf(rand.nextInt(nMax)));
					}
				}
				else {
					ind = replace(ind, pos, String.valueOf(rand.nextInt(nMax)));
				}				
				break;
			
			case '(':
				currDepth++;
				pos++;
				break;
			case ')':
				currDepth--;
				pos++;
				break;
				
			default:
				pos++;
				break;
			}
		}while(ind.charAt(pos) != ';');
		
		return ind;
	}
	
	/**
	 * Creates a child in the form (C ... C) including n times the 'C'
	 * 
	 * @param n Number of children
	 * @return String
	 */
	private String child(int n) {
		String child = "(";
		for(int i=0; i<n; i++) {
			child += "C ";
		}
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
		int currNChild = rand.nextInt(maxChildren-1)+2; //between 2 and max, included
		return child(currNChild);
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
