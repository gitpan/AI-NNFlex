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
=pod

=head1 AI::NNFlex::Dataset::new

=item 

Constructor, creates a data set to run through the network.
Takes as parameters an anonymous hash containing input array
and target output array, as follows:

my $dataset = AI::NNFlex::Dataset->new([
		[0,1,1,0,1,0],[1,1],
		[0,0,1,1,0,0],[1,0]]);

returns dataset object

=cut

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
=pod

=head1 AI::NNFlex::Datasets::run

=item

Run a data set through the network and return the outputs.

Takes as parameter the network object you want to run the
data against.

Inspired by the Xerion/UTS approach to managing datasets.

Syntax:

$dataset->run($network);

returns a reference to an hash containing input to actual
outputs in array form.

=cut
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
=pod

=head1 AI::NNFlex::Dataset::learn

for each entry in the dataset, run through the network then call
the networks learning algorithm. Takes as a parameter the network
you want to train.

Syntax:

my $error = $dataset->learn($network);

returns network sqrd err.

=cut

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

