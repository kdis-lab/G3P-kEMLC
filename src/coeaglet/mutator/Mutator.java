package coeaglet.mutator;

import java.util.ArrayList;
import java.util.Collections;

import net.sf.jclec.ISpecies;
import net.sf.jclec.listind.MultipListGenotype;
import net.sf.jclec.listind.MultipListIndividual;
import net.sf.jclec.listind.MultipListMutator;
import net.sf.jclec.listind.MultipListSpecies;

/**
 * Class implementing the mutator
 * 
 * @author Jose M. Moyano
 *
 */
public class Mutator extends MultipListMutator {

	/** Serialization constant */
	private static final long serialVersionUID = 2455293830055566959L;
	
	/** Individual species (taked from execution context) */
	protected transient MultipListSpecies species;
	
	/** Genotype schema */ 	
	protected transient MultipListGenotype schema;
	
	/** Max int value for the list */
	public int maxInt;

	
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
	 * Set the max int value for the list
	 * @param maxInt
	 */
	public void setMaxInt(int maxInt) {
		this.maxInt = maxInt;
	}
	
	/**
	 * Mutate next individual
	 * 
	 * This mutator randomly changes one value in the list for another random value
	 */
	protected void mutateNext() {
		//Get individual to be mutated
		MultipListIndividual mutant = (MultipListIndividual) parentsBuffer.get(parentsCounter);
		
		//Genotype for mutated individual
		ArrayList<Integer> gen = new ArrayList<Integer>(mutant.getGenotype().genotype);
		
		/*
		 * r1: gene of the individual to remove
		 * r2: new gene to include in the individual
		 */
		int r1, r2;
		r1 = randgen.choose(0, gen.size());
		do {
			r2 = randgen.choose(0, maxInt);
		}while(gen.contains(r2) || r2==r1);

		//In position r1, set gene r2
		gen.set(r1, r2);
		Collections.sort(gen);
		
		sonsBuffer.add(species.createIndividual(new MultipListGenotype(mutant.getGenotype().subpop, gen)));
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
