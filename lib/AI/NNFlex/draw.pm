##########################################################
# AI::NNFlex::draw
##########################################################
# Draw a gif file of the network
##########################################################
# 
##########################################################
# Versions
# ========
#
# 1.0   20041112       CColbourn       New module
# 1.1	20050115	CColbourn	Added PNG support
#
###########################################################
# ToDo
# ----
# STILL IN DEVELOPMENT
#
###########################################################
#
use strict;
package AI::NNFlex::draw;

=pod
=head1 AI::NNFlex::draw::network

simple network diagram routine.
syntax:

	my $gif = AI::NNFlex::draw->network($network,{'type'=>'png/GIF'});

Returns a GIF/png object. This is the first tentative step 
towards a GUI control for the bundle;

NB: the diagram returned displays the activation of the node
as a red circle inside the green circle.
Copyright (c) 2004-2005 Charles Colbourn. All rights reserved. This program is free software; you can redistribute it and/or modify
it under the same terms as Perl itself.
=cut
###############################################################
# 
#
###############################################################
use strict;
use GD;

sub network
{
	shift;
	my $network = shift;
	my $params = shift;
	my %config = %$params;

	my @nodeList;

	$network->dbug("Entered AI::NNFlex::draw with network $network",6);

	# I nicked this code from my ldapmonitor script, so there could
	# be ldap variables lurking in here


	my $numberOfLayers = scalar @{$network->{'layers'}};

	my $maxNodes;
	foreach (@{$network->{'layers'}})
	{
		if (scalar @{$_->{'nodes'}} > $maxNodes)
		{
			$maxNodes = scalar @{$_->{'nodes'}};
		}
	}


	# Find out how many nodes in each layer, and allow space for the most
        my ($imageHeight,$imageHeightFactor,$imageWidth,$imageWidthFactor);
	$imageHeightFactor = $maxNodes;

        $imageHeight +=40; # to allow 20 pixels around the outside
        $imageHeight += $imageHeightFactor*40; # to allow 40 pixels per node
        $imageHeight += (80 * ($imageHeightFactor -1)); # allow 60 pixels between nodes


	$imageWidthFactor = $numberOfLayers;
	# space the layers out more when there are large numbers of nodes
	# otherwise fill doesn't work properly and you end up with a solid
	# block of colour
	$imageWidthFactor = $imageWidthFactor *($imageHeightFactor/4);
        $imageWidth +=40; # to allow 20 pixels around the outside
        $imageWidth += $imageWidthFactor*40; # to allow 40 pixels per node
        $imageWidth += (80 * ($imageWidthFactor -1)); # allow 60 pixels between nodes
	


	$network->dbug("Image height=$imageHeight, width=$imageWidth",6);

        my  $image = new GD::Image($imageWidth,$imageHeight);
        my $white = $image->colorAllocate(255,255,255);
        my $black = $image->colorAllocate(0,0,0);
        my $red = $image->colorAllocate(255,0,0);
        my $green = $image->colorAllocate(51,204,102);
	my $grey = $image->colorAllocate(190,190,190);
        $image->transparent($white);


        
        # Allocate a location to each node 
        my %nodeCoords;
        my $currentLocationX = 40;
        my $currentLocationY = 0;


	foreach my $layer(@{$network->{'layers'}})
        {
		my $divisor = scalar (@{$layer->{'nodes'}});
		foreach my $node (@{$layer->{'nodes'}})
		{
                	$currentLocationY +=  (($imageHeight)/($divisor+1));
			$nodeCoords{$node} = [$currentLocationX,$currentLocationY];
			push @nodeList,$node;
		}
		#$currentLocationX += 60;
		$currentLocationX += 60 * ($imageWidthFactor/$numberOfLayers);
		$currentLocationY=0;

	}


        # Print the nodes on the picture

        foreach my $node (@nodeList)
        {
		$network->dbug("Drawing node $node at ${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1]",6);
		if ($node->{'active'})
		{
			$image->arc(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],40,40,0,360,$green);
			$image->fill(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],$green);
		}
		else
		{
			$image->arc(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],40,40,0,360,$grey);
			$image->fill(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],$grey);
		}

        }


        foreach my $node (@nodeList)
	{
		foreach my $connectedNode (@{$node->{'connectedNodesWest'}->{'nodes'}})       
		{
			if ($node->{'active'} && $connectedNode->{'active'})
			{
				#Forget the weights for the moment
				$image->line(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],${$nodeCoords{$connectedNode}}[0], ${$nodeCoords{$connectedNode}}[1],$green);
			}
		}
	}

	# apply the activation
	foreach my $node (@nodeList)
	{
		my $activation;
		$activation = $node->{'activation'};
		$network->dbug("Node $node activation $activation",6);
		if ($activation > 1){$activation = 1};
		if ($activation < 0){$activation = 0};
		if ($node->{'active'})
		{
			$network->dbug("Node $node adjusted activation $activation",6);
 	               $image->arc(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],40*$activation,40*$activation,0,360,$red);
        	        $image->fillToBorder(${$nodeCoords{$node}}[0],${$nodeCoords{$node}}[1],$red,$red);
		}
	}

	$network->dbug("Returning image $image",6);

	if (uc($config{'type'}) eq "GIF"){return $image->gif};
	if (uc($config{'type'}) eq "PNG"){return $image->png};

	# return gif by default
	return $image->gif;
}


1;


