package net.sf.jclec.listind;

import net.sf.jclec.ISpecies;

/**
 * Abstract implementation for MultipListSpecies.
 * 
 * This class  contains a MultipListGenotype that contains the genotype schema for all
 * represented individuals. This schema can be set in a subclass of this or can 
 * be calculated from other problem information.
 * 
 * It also allows the use of multiple subpopulations.
 * 
 * @author Jose M. Moyano
 * @author Sebastian Ventura
 * 
 * @see MultipListIndividualSpecies
 */

@SuppressWarnings("serial")
public abstract class MultipListSpecies implements ISpecies
{
	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------------Properties
	/////////////////////////////////////////////////////////////////
	
	/** Genotype schema */
	protected MultipListGenotype genotypeSchema;
	
	/**
	 * Identifier of subpopulation
	 */
	protected int p;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////
	
	/**
	 * Empty constructor
	 */
	public MultipListSpecies() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// Factory method
	
	/**
	 * Factory method.
	 * 
	 * @param genotype Individual genotype.
	 * 
	 * @return A new instance of represented class
	 */
	public abstract MultipListIndividual createIndividual(MultipListGenotype genotype);	

	// Genotype information

	/**
	 * Informs about individual genotype length.
	 * 
	 * @return getGenotypeSchema().length
	 */
	public int getGenotypeLength() 
	{
		return genotypeSchema.genotype.size();
	}

	/**
	 * @return This genotype schema
	 */
	public MultipListGenotype getGenotypeSchema() 
	{
		return genotypeSchema;
	}
	
	/**
	 * Get identifier of subpopulation
	 * 
	 * @return Identifier of subpopulation
	 */
	public int getSubpopId() {
		return p;
	}
}
