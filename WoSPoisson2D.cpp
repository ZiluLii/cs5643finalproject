// Quick and dirty implementation of a 2D Poisson solver via random walks.
// Corresponds to the naïve estimator given in Equation 8 of
// Sawhney & Crane, "Monte Carlo Geometry Processing" (2020).
// NOTE: this code makes a few shortcuts for the sake of code brevity; may
// be more suitable for tutorials than for production code/evaluation.
// To compile: c++ -std=c++17 -O3 -pedantic -Wall WoSPoisson2D.cpp -o poisson
#include <algorithm>
#include <array>
#include <complex>
#include <functional>
#include <iostream>
#include <random>
#include <vector>
#include <fstream>
using namespace std;

// use std::complex to implement 2D vectors
using Vec2D = complex<float>;
float dot(Vec2D u, Vec2D v) { return real(conj(u)*v); }
float length( Vec2D u ) { return sqrt( norm(u) ); }

// a segment is just a pair of points
using Segment = array<Vec2D,2>;

// returns the point on segment s closest to x
Vec2D closestPoint( Vec2D x, Segment s ) {
   Vec2D u = s[1]-s[0];
   float t = clamp(dot(x-s[0],u)/dot(u,u),0.f,1.f);
   return (1-t)*s[0] + t*s[1];
}

// returns a random value in the range [rMin,rMax]
float random( float rMin, float rMax ) {
   const float rRandMax = 1./(float)RAND_MAX;
   float u = rRandMax*(float)rand();
   return u*(rMax-rMin) + rMin;
}

// harmonic Green's function for a 2D ball of radius R
float G( float r, float R )
{
   float GrR = log(R/r)/(2.*M_PI);
   if( isnan(GrR) ) return 0;
   return GrR;
}

float solve3(Vec2D x0, // evaluation point
             vector<Segment> segments, // geometry
             function<float(Vec2D)> f, // source term
             function<float(Vec2D)> g  // boundary conditions
           ) {
   const float eps = 0.01; // stopping tolerance
   const int nWalks = 8; // number of Monte Carlo samples
   const int maxSteps = 16; // maximum walk length

   float sum = 0.;
   // Will vectorize this loop
   for( int i = 0; i < nWalks; i++ ) {
      Vec2D x = x0;
      float R;
      for (int steps = 0; steps < maxSteps; steps++) {
         R = numeric_limits<float>::max();
         for( auto s : segments ) {
            Vec2D p = closestPoint( x, s );
            R = min( R, length(x-p) );
         }
         if (R > eps) {
            //new stuff for Poisson
            // sample a point y uniformly from the ball of radius R around x
            float r = R*sqrt(random(0.,1.));
            float alpha = random( 0., 2.*M_PI );
            Vec2D y = x + Vec2D( r*cos(alpha), r*sin(alpha) );
            sum += (M_PI*R*R)*f(y)*G(r,R);
             
            // sample the next point x uniformly from the sphere around x
            float theta = random( 0., 2.*M_PI );
            x = x + Vec2D( R*cos(theta), R*sin(theta) );
         }
      }
      sum += g(x);
   }
   return sum / nWalks; // Monte Carlo estimate
}

float solve2(Vec2D x0, // evaluation point
             vector<Segment> segments, // geometry
             function<float(Vec2D)> f, // source term
             function<float(Vec2D)> g  // boundary conditions
           ) {
   const float eps = 0.01; // stopping tolerance
   const int nWalks = 128; // number of Monte Carlo samples
   const int maxSteps = 4; // maximum walk length

   float sum = 0.;
   // Will vectorize this loop
   for( int i = 0; i < nWalks; i++ ) {
      Vec2D x = x0;
      float R;
      for (int steps = 0; steps < maxSteps; steps++) {
         R = numeric_limits<float>::max();
         for( auto s : segments ) {
            Vec2D p = closestPoint( x, s );
            R = min( R, length(x-p) );
         }
         if (R > eps) {
            //new stuff for Poisson
            // sample a point y uniformly from the ball of radius R around x
            float r = R*sqrt(random(0.,1.));
            float alpha = random( 0., 2.*M_PI );
            Vec2D y = x + Vec2D( r*cos(alpha), r*sin(alpha) );
            sum += (M_PI*R*R)*f(y)*G(r,R);
             
            // sample the next point x uniformly from the sphere around x
            float theta = random( 0., 2.*M_PI );
            x = x + Vec2D( R*cos(theta), R*sin(theta) );
         }
      }
      sum += g(x);
   }
   return sum / nWalks; // Monte Carlo estimate
}

// solves a Laplace equation Δu = f at x0, where the boundary is given
// by a collection of segments, and the boundary conditions are given
// by a function g that can be evaluated at any point in space
float solve( Vec2D x0, // evaluation point
             vector<Segment> segments, // geometry
             function<float(Vec2D)> f, // source term
             function<float(Vec2D)> g  // boundary conditions
           ) {
   // const float eps = 0.001; // stopping tolerance
   const float eps = 0.01; // stopping tolerance
   // const int nWalks = 32; // number of Monte Carlo samples
   // const int nWalks = 256; // number of Monte Carlo samples
   const int nWalks = 1024; // number of Monte Carlo samples
   const int maxSteps = 64; // maximum walk length

   float sum = 0.;
   for( int i = 0; i < nWalks; i++ ) {
      Vec2D x = x0;
      float R;
      int steps = 0;
      do {

         // get the distance to the closest point on any segment
         R = numeric_limits<float>::max();
         for( auto s : segments ) {
            Vec2D p = closestPoint( x, s );
            R = min( R, length(x-p) );
         }

         // sample a point y uniformly from the ball of radius R around x
         float r = R*sqrt(random(0.,1.));
         float alpha = random( 0., 2.*M_PI );
         Vec2D y = x + Vec2D( r*cos(alpha), r*sin(alpha) );
         sum += (M_PI*R*R)*f(y)*G(r,R);

         // sample the next point x uniformly from the sphere around x
         float theta = random( 0., 2.*M_PI );
         x = x + Vec2D( R*cos(theta), R*sin(theta) );
         steps++;
      }
      while( R > eps && steps < maxSteps );

      sum += g(x);
   }

   return sum/nWalks; // Monte Carlo estimate
}

// reference solution
float uref( Vec2D x )
{
   return cos(2.f*M_PI*real(x)) * sin(2.f*M_PI*imag(x));
}

// Laplacian of reference solution
float laplace_uref( Vec2D x )
{
   return 8.f*(M_PI*M_PI) * cos(2.f*M_PI*real(x)) * sin(2.f*M_PI*imag(x));
}

// four segments enclosing the unit square
vector<Segment> scene = {
   {{ Vec2D(0.0, 0.0), Vec2D(1.0, 0.0) }},
   {{ Vec2D(1.0, 0.0), Vec2D(1.0, 1.0) }},
   {{ Vec2D(1.0, 1.0), Vec2D(0.0, 1.0) }},
   {{ Vec2D(0.0, 1.0), Vec2D(0.0, 0.0) }}
};

int main( int argc, char** argv )
{
   srand( time(NULL) );

   // ofstream out( "WoSPoisson-2D.csv" );
   // ofstream out( "WoSPoisson-2D-2.csv" );
   ofstream out( "try4-WoSPoisson.csv" );
   // To validate the implementation we solve the Poisson equation
   //
   //    Δu = Δu0  on Ω
   //     u =  u0  on ∂Ω
   //
   // where u0 is some reference function.  The solution should
   // converge to u = u0 as the number of samples N increases.

   int s = 128;
   for( int j = 0; j < s; j++ )
   {
      cerr << "row " << j << " of " << s << endl;
      for( int i = 0; i < s; i++ )
      {
         Vec2D x0( (float)i/(float)s, (float)j/(float)s );
         // float u = solve( x0, scene, laplace_uref, uref );
         float u = solve2( x0, scene, laplace_uref, uref );
         out << u;
         if( i < s-1 ) out << ",";
      }
      out << endl;
   }
   return 0;
}
