package net.sf.jclec.stringtree;

import net.sf.jclec.ISpecies;

import net.sf.jclec.base.AbstractMutator;

/**
 * BinArrayIndividual (and subclasses) specific mutator.  
 * 
 * @author Sebastian Ventura
 */

public abstract class StringTreeMutator extends AbstractMutator 
{
	private static final long serialVersionUID = -2598591683615873232L;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------ Operation variables
	/////////////////////////////////////////////////////////////////

	/** Individual species (taked from execution context) */
	
	protected transient StringTreeSpecies species;
	
	/** Genotype schema */ 
	
	protected transient String schema;
	
	/////////////////////////////////////////////////////////////////
	// ------------------------------------------------- Constructors
	/////////////////////////////////////////////////////////////////

	/**
	 * Empty (default) constructor.
	 */
	
	public StringTreeMutator() 
	{
		super();
	}

	/////////////////////////////////////////////////////////////////
	// ----------------------------------------------- Public methods
	/////////////////////////////////////////////////////////////////
	
	// AbstractMutator methods
	
	/**
	 * {@inheritDoc}
	 */
	
	@Override
	protected void prepareMutation() 
	{
		ISpecies species = context.getSpecies();
		if (species instanceof StringTreeSpecies) {
			// Set individuals species
			this.species = (StringTreeSpecies) species;
			// Sets genotype schema
			this.schema = this.species.getGenotypeSchema();
		}
		else {
			throw new IllegalStateException("Invalid species in context");
		}
	}

	/* 
	 * Este mtodo fija el schema que vamos a utilizar para mutar los genotipos
	 * de los nuevos individuos. Para ello, asegura que el objeto species que
	 * representa a los individuos de la poblacin es de tipo IBinArraySpecies.
	 * En caso negativo, lanza una excepcin.
	 */

	/////////////////////////////////////////////////////////////////
	// -------------------------------------------- Protected methods
	/////////////////////////////////////////////////////////////////

}
