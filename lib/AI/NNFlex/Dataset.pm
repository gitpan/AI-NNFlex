##########################################################
# AI::NNFlex::Dataset
##########################################################
# Dataset methods for AI::NNFlex - perform learning etc
# on groups of data
#
##########################################################
# Versions
# ========
#
# 1.0	20050115	CColbourn	New module
#
##########################################################
# ToDo
# ----
#
#
###########################################################
#
use strict;
package AI::NNFlex::Dataset;


###########################################################
# AI::NNFlex::Dataset::new
###########################################################
sub new
{
	my $class = shift;
	my $params = shift;

	my %attributes;
	$attributes{'data'} = $params;

	my $dataset = \%attributes;
	bless $dataset;
	return $dataset;
}


###########################################################
# AI::NNFlex::Datasets::run
###########################################################
sub run
{
	my $self = shift;
	my $network = shift;
	my @outputs;
	my $counter=0;

	for (my $itemCounter=0;$itemCounter<(scalar @{$self->{'data'}});$itemCounter +=2)
	{
		$network->run(@{$self->{'data'}}[$itemCounter]);
		$outputs[$counter] = $network->output();
		$counter++;
	}

	return \@outputs;

}

###############################################################
# AI::NNFlex::Dataset::learn
###############################################################
sub learn
{
	my $self = shift;
	my $network = shift;
	my $error;

	for (my $itemCounter=0;$itemCounter<(scalar @{$self->{'data'}});$itemCounter +=2)
	{
		$network->run(@{$self->{'data'}}[$itemCounter]);
		$error += $network->learn(@{$self->{'data'}}[$itemCounter+1]);
	}

	$error = $error*$error;

	return $error;
}

1;

