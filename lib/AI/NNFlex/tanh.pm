##########################################################
# AI::NNFlex::tanh
##########################################################
# tanh activation function for NNFlex
##########################################################
# This code copied pretty much wholesale from Phil Brierleys
# example - see the NNFlex readme
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
=head1 AI::NNFlex::tanh

=item

sigmoid activation function. This code is imported into
the NNFlex namespace during init

syntax:

        my $value = $network->tanh(<value>);

Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.

=cut
############################################################
	
sub tanh
{

	my $network = shift;
	my $value = shift;

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
	$network->dbug("Tanh activation returning $value",5);	
	return $value;
}

1;
