##########################################################
# AI::NNFlex::backprop
##########################################################
# This is the first learning module for NNFlex
#
##########################################################
# Versions
# ========
#
# 1.0	20041018	CColbourn	New module
# 1.1	20050116	CColbourn	added dataset translater
#					to ::learn
# 1.2	20050201	CColbourn	Inseted call to activation
#					slope function
#
##########################################################
# ToDo
# ----
#
#
###########################################################
#
use strict;

###########################################################
# AI::NNFlex::backprop::learn
###########################################################
=pod

=head1 AI::NNFlex::backprop

=item

This module is the backpropagation algorithm for NNFlex. it
is included in the NNFlex namespace at run time. See documentation
below for standard methods.

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut

###########################################################
#AI::NNFlex::backprop::learn
###########################################################
=pod

=head1 AI::NNFlex::backprop::learn

=item

Takes as a parameter a reference to the desired output
pattern, performs one learning pass back through the network 
with normal backprop procedures to bring the network closer 
to convergence.

This package is imported into the NNFlex namespace at runtime
via a parameter to the network object.

syntax:
 $network->learn([0,1,1,0]);

=cut

###########################################################
# AI::NNFlex::backprop::calc_error
###########################################################
=pod
=cut

sub calc_error
{
	my $network = shift;

	my $outputPatternRef = shift;
	my @outputPattern = @$outputPatternRef;

	my @debug = @{$network->{'debug'}};

	if (scalar @debug > 0)
	{$network->dbug ("Output pattern @outputPattern received by backprop",4);}


	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	if (scalar @$outputLayer != scalar @outputPattern)
	{	
		$network->dbug ("Wrong number of output values, net has ".scalar @$outputLayer." nodes",0);
		return 0;
	}

	# Now calculate the error
	my $counter=0;
	foreach (@$outputLayer)
	{	
		my $value = $_->{'activation'} - $outputPattern[$counter];
		
		$_->{'error'} = $value;
		$counter++;
		if (scalar @debug > 0)
		{$network->dbug ("Error on output node $_ = ".$_->{'error'},4);}
	}


}

########################################################
# AI::NNFlex::backprop::learn
########################################################
=pod

=head1 AI::NNFlex::backprop::learn

=item

learn is the main method of the backprop module. It calls
calc_error to calculate the output error, calls output_adjust
to adjust the weights from the last hidden to the output
layer, then calls hidden_adjust to adjust the weights to the
hidden layers
finally it returns the network sqrd error.

=cut
#########################################################
sub learn
{

	my $network = shift;

	my $outputPatternRef = shift;

	# if this is an incorrect dataset call translate it
	if ($outputPatternRef =~/Dataset/)
	{
		return ($outputPatternRef->learn($network))
	}



	my @outputPattern = @$outputPatternRef;

	$network->calc_error($outputPatternRef);

	#calculate & apply dWs
	$network->hiddenToOutput;
	if (scalar @{$network->{'layers'}} > 2) 
	{
		$network->hiddenOrInputToHidden;
	}

	# calculate network sqErr
	my $Err = $network->RMSErr($outputPatternRef);
	return $Err;	
}


#########################################################
# AI::NNFlex::backprop::hiddenToOutput
#########################################################
=pod

=head1 AI::NNFlex::backprop::hiddenToOutput

=item
Performs weight changes for all nodes in the output layer
nodes 'connectedNodesWest' attributes, based on backprop
Output weights delta

=cut
##########################################################
sub hiddenToOutput
{
	my $network = shift;

	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	my @debug = @{$network->{'debug'}};

	foreach my $node (@$outputLayer)
	{
		my $connectedNodeCounter=0;
		foreach my $connectedNode (@{$node->{'connectedNodesWest'}->{'nodes'}})
		{
			if (scalar @debug > 0)
			{$network->dbug("Learning rate is ".$network->{'learningrate'},4);} 
			my $deltaW = (($network->{'learningrate'}) * ($node->{'error'}) * ($connectedNode->{'activation'}));
			if (scalar @debug > 0)
			{$network->dbug("Applying delta $deltaW on hiddenToOutput $connectedNode to $node",4);}
			# 
			$node->{'connectedNodesWest'}->{'weights'}->[$connectedNodeCounter] -= $deltaW;
			$connectedNodeCounter++;	
		}
	}
}

######################################################
# AI::NNFlex::backprop::hiddenOrInputToHidden
######################################################
=pod

=head1 AI::NNFlex::backprop::hiddenOrInputToHidden

=item
This subroutine calculates and applies delta weights that are not directly
derived from the output layer. I.e. if you have a 2 layer network this
will never get called.

If you have a 3 layer network this will be called to delta the weights
between input and hidden layers.

If you have a 4 layer network it will be called to delta the weights between
the two hidden layers, and the input to hidden layer

=cut

#######################################################
sub hiddenOrInputToHidden
{

	my $network = shift;

	my @layers = @{$network->{'layers'}};
	my @debug =@{$network->{'debug'}};
	

	# remove the last element (The output layer) from the stack
	# because we've already calculated dW on that
	pop @layers;

	if (scalar @debug > 0)
	{$network->dbug("Starting backprop of error on ".scalar @layers." hidden layers",4);}

	foreach my $layer (reverse @layers)
	{
		foreach my $node (@{$layer->{'nodes'}})
		{
			if (!$node->{'connectedNodesWest'}) {last}

			my $nodeError;
			my $connectedNodeCounter=0;
			foreach my $connectedNode (@{$node->{'connectedNodesEast'}->{'nodes'}})
			{
				$nodeError += ($connectedNode->{'error'}) * ($connectedNode->{'connectedNodesWest'}->{'weights'}[$connectedNodeCounter]);
			}
			if (scalar @debug > 0)
			{$network->dbug("Hidden node $node error = $nodeError",4);}
			$node->{'error'} = $nodeError;


			# update the weights from nodes inputting to here
			my $connectedNodeCounter=0;
			foreach my $westNodes (@{$node->{'connectedNodesWest'}->{'nodes'}})
			{
				# get the slope from the activation function component
				my $value = $node->{'activation'};
				my $slope = $node->{'activationfunction'}."_slope";
				my $evalstring = "\$value = \$network->$slope(\$value);";
				eval($evalstring);

				$value = $value * $node->{'error'} * $network->{'learningrate'} * $westNodes->{'activation'};
				
				my $dW = $value;
				if (scalar @debug > 0)
				{$network->dbug("Applying deltaW $dW to inputToHidden connection from $westNodes to $node",4);}
				$node->{'connectedNodesWest'}->{'weights'}->[$connectedNodeCounter] -= $dW;
				$connectedNodeCounter++;
				if (scalar @debug > 0)
				{$network->dbug("Weight now ".$node->{'connectedNodesWest'}->{'weights'}->[$connectedNodeCounter],4);}
			}	


		}
	}
								
				

}

#########################################################
# AI::NNFlex::backprop::sqrErr
#########################################################
=pod

=head1 AI::NNFlex::backprop::sqrErr

=item

Calculates the network squared error after a single
backprop pass. Internal to ::backprop

=cut
##########################################################
sub RMSErr
{
	my $network = shift;

	my $outputPatternRef = shift;
	my @outputPattern = @$outputPatternRef;

	my @debug = @{$network->{'debug'}};

	my $sqrErr;

	my $outputLayer = $network->{'layers'}->[-1]->{'nodes'};

	if (scalar @$outputLayer != scalar @outputPattern)
	{	
		$network->dbug("Wrong number of output values, net has ".scalar @$outputLayer." nodes",0);
		return 0;
	}

	# Now calculate the error
	my $counter=0;
	foreach (@$outputLayer)
	{	
		my $value = $_->{'activation'} - $outputPattern[$counter];

		$sqrErr += $value *$value;
		$counter++;
		if (scalar @debug > 0)
		{$network->dbug("Error on output node $_ = ".$_->{'error'},4);}
	}

	my $error = sqrt($sqrErr);

	return $error;
}

1;

