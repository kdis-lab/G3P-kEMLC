import java.util.Random;

public class Main {
	
	/**
	 * Random numbers generator
	 */
	static Random rand = new Random();
	
	/**
	 * Maximum depth of individuals
	 */
	static int maxDepth = 3;
	
	/**
	 * Number of childs at each node
	 */
	static int nChilds = 4;
	
	public static void main(String[] args) {
		int nMax = 120;	
		Utils utils = new Utils();
		
		IndividualCreator creator = new IndividualCreator();
		
		int iters = 1000;
		for(int i=0; i<iters; i++) {
			String ind1 = creator.create(nMax, maxDepth, nChilds);
			String ind2 = creator.create(nMax, maxDepth, nChilds);
			
			System.out.println(ind1);
			System.out.println(ind2);
			if(!utils.checkInd(ind1, nChilds)) {
				System.out.println("ind not feasible: " + ind1);
				System.exit(1);
			}
			if(!utils.checkInd(ind2, nChilds)) {
				System.out.println("ind not feasible: " + ind2);
				System.exit(1);
			}
			
			Crossover c = new Crossover();
			c.setRand(rand);
			c.setMaxTreeDepth(maxDepth);
			String [] crossed = c.cross(ind1, ind2);
			System.out.println("c1: " + crossed[0]);
			System.out.println("c2: " + crossed[1]);
			if(!utils.checkInd(crossed[0], nChilds)) {
				System.out.println("crossed not feasible: " + crossed[0]);
				System.exit(1);
			}
			if(!utils.checkInd(crossed[1], nChilds)) {
				System.out.println("crossed not feasible: " + crossed[1]);
				System.exit(1);
			}
			System.out.println();
			
			Mutator m = new Mutator();
			m.setMaxTreeDepth(maxDepth);
			m.setnMax(nMax);
			m.setnChilds(nChilds);
			String mut1 = m.mutate(ind1);
			String mut2 = m.mutate(ind2);
			System.out.println("m1: " + mut1);
			System.out.println("m2: " + mut2);
			System.out.println("---");
			System.out.println();
			if(!utils.checkInd(mut2, nChilds)) {
				System.out.println("mutated not feasible: " + mut2);
				System.exit(1);
			}
			if(!utils.checkInd(mut2, nChilds)) {
				System.out.println("mutated not feasible: " + mut2);
				System.exit(1);
			}
		}
	}
}
