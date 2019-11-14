package coeaglet.mutator;

import java.util.ArrayList;

import net.sf.jclec.ISpecies;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.listind.MultipListMutator;
import net.sf.jclec.listind.MultipListSpecies;
import net.sf.jclec.util.random.IRandGen;

/**
 * Class implementing the mutator
 * 
 * @author Jose M. Moyano
 *
 */
public class SubpopMutator extends MultipListMutator {

	/** Serialization constant */
	private static final long serialVersionUID = 2455293830055566959L;
	
	/** Individual species (taked from execution context) */
	protected transient MultipListSpecies species;
	
	/** Genotype schema */ 	
	protected transient MultipListGenotype schema;
	
	/** Max int value for the list */
	public int nSubpops;

	
	/**
	 * Constructor
	 */
	public SubpopMutator()
	{
		super();
	}
	
	@Override
	public boolean equals(Object other)
	{
		if (other instanceof SubpopMutator) {
			return true;
		}
		else {
			return false;
		}
	}
	
	/**
	 * Set the max int value for the list
	 * @param maxInt
	 */
	public void setNSubpops(int nSubpops) {
		this.nSubpops = nSubpops;
	}
	
	/**
	 * Mutate next individual
	 * 
	 * This mutator randomly changes one value in the list for another random value
	 */
	protected void mutateNext() {
		//Get individual to be mutated
		MultipListIndividual mutant = (MultipListIndividual) parentsBuffer.get(parentsCounter);
		
		//Return individual with same genotype but new subpopulation
		sonsBuffer.add(mutateInd(mutant, nSubpops, randgen));
	}
	
	
	/**
	 * Mutate a given individual. Changes its subpopulation.
	 * 
	 * @param ind Individual to mutate
	 * @return Mutated individual
	 */
	public MultipListIndividual mutateInd(MultipListIndividual mutant, int nSubpops, IRandGen randgen) {
		//Genotype for mutated individual
		ArrayList<Integer> gen = new ArrayList<Integer>(mutant.getGenotype().genotype);
		int currentSubpop = mutant.getSubpop();
				
		/*
		 * r: new subpop
		 */
		int r;
		do {
			r = randgen.choose(0, nSubpops);
		}while(r == currentSubpop);
		
		MultipListIndividual mutInd = new MultipListIndividual(new MultipListGenotype(r, gen));
		
		return mutInd;
	}
	
	@Override
	protected void prepareMutation() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof MultipListSpecies) {
			// Set individuals species
			this.species = (MultipListSpecies) species;
			// Sets genotype schema
			this.schema = this.species.getGenotypeSchema();
		}
		else {
			throw new IllegalStateException("Invalid species in context");
		}
	}
	
}
