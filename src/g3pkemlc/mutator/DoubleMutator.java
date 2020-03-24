package g3pkemlc.mutator;

import java.util.ArrayList;
import java.util.Collections;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class DoubleMutator extends Mutator {

	/**
	 * 
	 */
	private static final long serialVersionUID = -2361173927635315416L;

	double ratioCombinersMutation;
	
	public void setRatioCombinersMutation(double ratioCombinersMutation) {
		this.ratioCombinersMutation = ratioCombinersMutation;
	}
	
	@Override
	public String mutate(String ind) {
		if(randgen.coin()) {
			return super.mutate(ind);
		}
		else {
			//Count combiners
			Pattern p = Pattern.compile("[cv]");
			Matcher m = p.matcher(ind);
			int nCombiners = 0;
			
			//Increment the counter each time that a combiner is found
			while(m.find()){
				nCombiners++;
			}
			
			int nMutations = 0;
			if((nCombiners*ratioCombinersMutation) <= 1) {
				nMutations = 1;
			}
			else {
				nMutations = randgen.choose(1, (int)Math.round(nCombiners*ratioCombinersMutation));
			}
			
			System.out.println("nMut: " + nMutations + " / " + (int)Math.round(nCombiners*ratioCombinersMutation) + " / " + nCombiners);
			String mut = new String(ind);
			
			ArrayList<Integer> randomCombiners = new ArrayList<Integer>(nMutations);
			int r;
			while(randomCombiners.size() < nMutations) {
				r = randgen.choose(0, nCombiners);
				if(!randomCombiners.contains(r)) {
					randomCombiners.add(r);
				}
			}
			m = p.matcher(mut);
			for(int i=0; i<nCombiners; i++) {
				m.find();
				
				if(randomCombiners.contains(i)) {
					mut = changeCombiner(mut, m.start(), m.end());
				}
			}
			
			return mut;
		}
		
	}
	
	String changeCombiner(String ind, int combStart, int combEnd) {
		//Change the combiner type
		String newCombiner = "";
		if(ind.charAt(combStart) == 'c'){
			newCombiner = "v";
		}
		else if(ind.charAt(combStart) == 'v'){
			newCombiner = "c";
		}
		
		ind = ind.substring(0, combStart) + newCombiner + ind.substring(combEnd);
		
		return ind;
	}
}
