/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	// Trying out with 100 particles
	num_particles = 100;
	normal_distribution<double> dist_x(x, std[1]);
	normal_distribution<double> dist_y(y, std[2]);
	normal_distribution<double> dist_theta(theta, std[3]);

	weights.resize(num_particles, 1.0);
    default_random_engine gen;

	for (int i = 0; i < num_particles; ++i) {
        Particle particle;
        particle.id = i;
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
        particles.push_back(particle);
    }

    is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	// Most of the code taken from the lessons and slightly adapted here

	default_random_engine gen;

    for (int i = 0; i < num_particles; ++i) {
        Particle particle = particles[i];

		// No division by zero
        if (yaw_rate > 0.0001) {
            particle.x = particle.x + (velocity / yaw_rate) * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
            particle.y = particle.y + (velocity / yaw_rate) * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
            particle.theta = particle.theta + yaw_rate * delta_t;
        } else {
            particle.x = particle.x + velocity * cos(particle.theta) * delta_t;
            particle.y = particle.y + velocity * sin(particle.theta) * delta_t;
            particle.theta = particle.theta + yaw_rate * delta_t;
        }

        normal_distribution<double> dist_x(particle.x, std_pos[1]);
        normal_distribution<double> dist_y(particle.y, std_pos[2]);
        normal_distribution<double> dist_theta(particle.theta, std_pos[3]);

        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);

        particles[i] = particle;
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	vector<double> distances;

    for (int i = 0; i < predicted.size(); ++i) {

		// Let's set a really high number, something that could not happen in the real simulator.
        double min = 1000000.0;
        int min_index = 0;

        if (observations.size() > 0) {
            for (int j = 0; j < observations.size(); ++j) {
                double dis = dist(predicted[i].x, predicted[i].y, observations[j].x, observations[j].y);
                if (dis < min) {
                    min = dis;
                    min_index = j;
                }
            }
            observations[min_index].id = i;
        }
    }

}

// http://www.cplusplus.com/reference/random/normal_distribution/
// https://stackoverflow.com/questions/41538095/evaluate-multivariate-normal-gaussian-density-in-c
double calculateLikelihood(double x, double ux, double y, double uy, double sig_x, double sig_y){
	double gauss_norm = 1/(2* M_PI * sig_x * sig_y);
	double exponent = ( (x - ux) * (x - ux) / (2.0 * sig_x * sig_x) ) + ( (y - uy) * (y - uy) / (2.0 * sig_y * sig_y) );
	double likelihood = gauss_norm * exp(-exponent);
	return likelihood;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	for (int i = 0; i < num_particles; i++) {
		Particle &p = particles[i];
		double prob = 1;
		for (int j=0; j<observations.size(); j++){

			double obj_x = observations[j].x;
			double obj_y = observations[j].y;

			// Transformation from particle coordinate to Map coordinate system
			double trans_x = p.x + cos(p.theta) * obj_x - sin(p.theta) * obj_y;
			double trans_y = p.y + sin(p.theta) * obj_x + cos(p.theta) * obj_y;

			vector<Map::single_landmark_s> landmarks = map_landmarks.landmark_list;
			double min_dist = 1000000;
			double land_x = -1;
			double land_y = -1;

			for (int k=0; k < landmarks.size(); k++) {
				double act_x = landmarks[k].x_f;
				double act_y = landmarks[k].y_f;
				double dist = sqrt( (act_x -trans_x) * (act_x -trans_x) + (act_y - trans_y) * (act_y - trans_y) );

				if (dist < min_dist && dist <= sensor_range ) {
					min_dist = dist;
					land_x = act_x;
					land_y = act_y;
				}
			}

			// calculate likelihood using 2D - gaussian distribution
			// https://stackoverflow.com/questions/41538095/evaluate-multivariate-normal-gaussian-density-in-c
			double likelihood = calculateLikelihood(land_x, trans_x, land_y, trans_y, std_landmark[0], std_landmark[1]);
			prob *= likelihood;
		}
		p.weight = prob;
		weights[i] = prob;

	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(weights.begin(), weights.end());
    std::vector<Particle> resampled_particles;

    for(int n = 0; n < num_particles; ++n) {
        Particle particle = particles[d(gen)];
        resampled_particles.push_back(particle);
    }
    particles = resampled_particles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
