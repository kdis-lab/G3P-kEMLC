import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Class implementing several utilities to deal with GP-EMLC individuals
 * 
 * @author Jose M. Moyano
 *
 */
public class Utils {
	
	/**
	 * Random numbers generator
	 */
	public static Random rand = new Random();

	/**
	 * Empty constructor
	 */
	public Utils() {
		rand = new Random();
	}
	
	/**
	 * Constructor with random numbers generator
	 * 
	 * @param rand Random numbers generator
	 */
	public Utils(Random rand) {
		Utils.rand = rand;
	}
	
	/**
	 * Counts the parenthesis (therefore possible subtrees) of the tree
	 * 
	 * @param ind Individual
	 * @return Number of begin parenthesis
	 */
	public int countParenthesis(String ind) {
		return ((int) ind.chars().filter(ch -> ch == '(').count());
	}
	
	/**
	 * Counts the number of leaves in the individual
	 * 
	 * @param ind Individual
	 * @return Number of leaves
	 */
	public int countLeaves(String ind) {
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(ind);
		int count = 0;
		while(m.find()){
			count++;
		}
		
		return count;
	}
	
	/**
	 * Choose a random leaf
	 * 
	 * @param ind Individual
	 * @return Array with index of start and end of the current leaf
	 */
	public int[] chooseRandomLeaf(String ind) {
		int nLeaves = countLeaves(ind);
		int r = rand.nextInt(nLeaves);
		
		Pattern p = Pattern.compile("\\d+"); // "\d" is for digits in regex
		Matcher m = p.matcher(ind);
		int count = 0;
		do {
			m.find();
			count++;
		}while(count<=r);
		
		int[] leaf = new int[2];
		leaf[0] = m.start();
		leaf[1] = m.end();

		return leaf;
	}
	
	/**
	 * Calculate the depth of a given node in the tree
	 * 
	 * @param ind Individual
	 * @param nodeStart Index of the string where the node starts
	 * @return Depth of the node
	 */
	public int calculateNodeDepth(String ind, int nodeStart) {
		int depth = 0;
		
		for(int i=0; i<nodeStart; i++) {
			switch (ind.charAt(i)) {
			case '(':
				depth++;
				break;

			case ')':
				depth--;
				break;
			}
		}
		
		return depth;
	}
	
	/**
	 * Calculate the maximum depth of any node in the tree
	 * 
	 * @param tree Tree
	 * @return Maximum depth
	 */
	public int calculateTreeMaxDepth(String tree) {
		int currDepth = 0, maxDepth = -1;
		
		for(int i=0; i<tree.length(); i++) {
			switch (tree.charAt(i)) {
			case '(':
				currDepth++;
				if(currDepth > maxDepth) {
					maxDepth = currDepth;
				}
				break;

			case ')':
				currDepth--;
				break;
			}
		}
		
		return maxDepth;
	}
	
	/**
	 * Choose a random subtree of a given individual
	 * 
	 * @param ind Individual
	 * @param considerRoot True if the whole tree is considered as a feasible subtree and false otherwise
	 * @return Array with start and end indexes of the subtree
	 */
	public int[] chooseRandomSubTree(String ind, boolean considerRoot) {
		int parenthesis = countParenthesis(ind);
		int [] subTree = new int[2];
		int r;
		
		if(considerRoot || parenthesis > 1) {
			if(!considerRoot) {
				r = rand.nextInt(parenthesis-1)+1; //avoid to select root (parenthesis 0)
			}
			else {
				r = rand.nextInt(parenthesis);
			}
			
			
			int beginParenthesisCount = 0, extraBeginParenthesis = 0;
			int startSubstr=0, endSubstr=0;
			boolean extracting = false, found = false;
			int pos = 0;
			do {
				switch (ind.charAt(pos)) {
				case '(':
					if(!extracting) {
						if(beginParenthesisCount == r) {
							extracting = true;
							startSubstr = pos;
						}
						else {
							beginParenthesisCount++;
						}
					}
					else {
						extraBeginParenthesis++;
					}
					
					break;
				case ')':
					if(extracting) {
						if(extraBeginParenthesis > 0) {
							extraBeginParenthesis--;
						}
						else {
							endSubstr = pos;
							found = true;
						}
					}
					break;
					
				default:
					break;
				}
				pos++;
			}while(!found);
			
			subTree[0] = startSubstr;
			subTree[1] = endSubstr+1;
		}
		else {
			return null;
		}
		
		return subTree;
	}
	
	public boolean checkInd(String ind, int n) {
		Pattern pattern = Pattern.compile("\\((\\d+ )+\\d+\\)");
		
		while(pattern.matcher(ind).find()) {
			ind = ind.replaceAll("\\((\\d+ )+\\d+\\)", "0");
		}
		
		if(ind.equals("0;")) {
			return true;
		}
		else {
			return false;
		}
	}
	
}
