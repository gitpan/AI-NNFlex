##########################################################
# AI::NNFlex::linear
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
=head1 AI::NNFlex::linear

=item

Linear activation function. This code is imported into
the NNFlex namespace during init

syntax:
	my $value = $network->linear(<value>);

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut
############################################################

sub linear
{

	my $network = shift;
	my $value = shift;	
	$network->dbug("Linear activation returning $value",5);	
	return $value;
}

1;
