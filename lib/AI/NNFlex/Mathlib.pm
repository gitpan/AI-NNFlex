#######################################################
# AI::NNFlex::Mathlib
#######################################################
# Various custom mathematical functions for AI::NNFlex
#######################################################
#
# Version history
# ===============
#
# 1.0 	CColbourn	20050315	Compiled into a
#					single module
#
#######################################################
#Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify

package AI::NNFlex::Mathlib;
use strict;

#######################################################
# tanh activation function
#######################################################
sub tanh
{

	my $network = shift;
	my $value = shift;

	my @debug = @{$network->{'debug'}};

	my $a = exp($value);
	my $b = exp(-$value);
   if ($value > 20){ $value=1;}
    elsif ($value < -20){ $value= -1;}
    else
        {
	        my $a = exp($value);
       		 my $b = exp(-$value);
   	        $value =  ($a-$b)/($a+$b);
        }
	if (scalar @debug > 0)
	{$network->dbug("Tanh activation returning $value",5)};	
	return $value;
}

sub tanh_slope
{
	my $network = shift;
	my $value = shift;
	my @debug = @{$network->{'debug'}};


	my $return = 1-($value*$value);
	if (scalar @debug > 0)
	{$network->dbug("Tanh_slope returning $value",5);}

	return $return;
}

#################################################################
# Linear activation function
#################################################################
sub linear
{

	my $network = shift;
	my $value = shift;	

	my @debug = @{$network->{'debug'}};
	if (scalar @debug >0)
	{$network->dbug("Linear activation returning $value",5)};	
	return $value;
}

sub linear_slope
{
	my $network = shift;
	my $value = shift;
	my @debug = @{$network->{'debug'}};
	if (scalar @debug >0)
	{$network->dbug("Linear slope returning $value",5)};
	return $value;
}


############################################################
# P&B sigmoid activation (needs slope)
############################################################

sub sigmoid
{
	my $network = shift;
	my $value = shift;	
	$value = (1+exp(-$value))**-1;
	$network->dbug("Sigmoid activation returning $value",5);	
	return $value;
}

############################################################
# atanh error function
############################################################
sub atanh
{
	my $network = shift;
	my $value = shift;
	if ($value >-1 && $value <1)
	{
		$value = log((1+$value)/(1-$value))/2;
	}
	return $value;
}

1;

=pod

=head1 NAME

AI::NNFlex::Mathlib - miscellaneous mathematical functions for the AI::NNFlex NN package

=head1 DESCRIPTION

The AI::NNFlex::Mathlib package contains activation and error functions. At present there are the following:

Activation functions

=over

=item *
tanh

=item *
linear

=back

Error functions

=over

=item *
atanh

=back

If you want to implement your own activation/error functions, you can add them to this module. All activation functions require an additional function <function name>_slope, which returns the 1st order derivative of the function.


=head1 CHANGES


=head1 COPYRIGHT

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify it under the same terms as Perl itself.

=head1 CONTACT

 charlesc@nnflex.g0n.net



=cut
