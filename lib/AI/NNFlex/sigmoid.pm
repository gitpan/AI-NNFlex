##########################################################
# AI::NNFlex::sigmoid
##########################################################
# linear activation function for NNFlex
##########################################################
# 
##########################################################
# Versions
# ========
#
# 1.0   20040910        CColbourn       New module
#
###########################################################
# ToDo
# ----
#
#
###########################################################
#
use strict;

=pod
=head1 AI::NNFlex::sigmoid

=item

	Sigmoid activation function. This code is imported
	into the NNFlex namespace by an eval during
	$network->init;

	syntax:
		my $value = $network->sigmoid(<value>);


Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut
############################################################

sub sigmoid
{
	my $network = shift;
	my $value = shift;	
	$value = (1+exp(-$value))**-1;
	$network->dbug("Sigmoid activation returning $value",5);	
	return $value;
}
1;

