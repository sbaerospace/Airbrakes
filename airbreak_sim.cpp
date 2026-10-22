// Standard C++17 only. Build: c++ -std=c++17 -O2 airbrake_sim.cpp -o airbrake_sim
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>
#include <vector>
#include <limits>

namespace cfg {
constexpr double G = 9.80665;
constexpr double TARGET_M = 800.0 * 0.3048; // 243.84 m above launch point
constexpr double MASS_KG = 0.650;          // placeholder: constant coast mass
constexpr double AREA_M2 = 0.00385;       // fixed reference area (~70 mm diameter)
constexpr double BASE_CD = 0.50;          // placeholder, brakes retracted
// EXTRA effective Cd from brakes, all using AREA_M2 above. NOT measured data.
// Index 0 = retracted; index 9 = maximum extension. Total Cd = BASE_CD + entry.
constexpr std::array<double, 10> BRAKE_CD =
    {0.00, 0.08, 0.18, 0.32, 0.50, 0.72, 0.98, 1.28, 1.62, 2.00};
constexpr double K = 0.20;                 // settings per metre of apogee error
constexpr double INITIAL_IMPULSE_NS = 66.0;// ideal instantaneous NET momentum impulse
constexpr double INITIAL_HEIGHT_M = 0.0;
constexpr double PREDICTOR_RHO = 1.225;    // held CONSTANT in analytical predictor
constexpr double CONTROL_PERIOD_S = 0.25; // 4 Hz: choose a new command every 0.25 s
constexpr double SERVO_LEVELS_PER_S = 18.0;// 0 to 9 takes 0.5 s
constexpr double GUST_PERIOD_S = 0.20;
constexpr double GUST_TIME_CONSTANT_S = 0.60;
constexpr double MAX_TIME_S = 60.0;
}

// Physical conditions used by the numerical model. The predictor keeps nominal
// Cd and fixed rho, so these variations also test modest model mismatch.
struct Conditions {
    double densityScale = 1.0, dragScale = 1.0;
    double meanWind = 2.0, gustScale = 1.0;
};
struct Options {
    double dt = 0.001, gain = cfg::K, impulse = cfg::INITIAL_IMPULSE_NS;
    std::uint32_t seed = 42;
    int runs = 1;
    bool disturbances = true, predictBase = false, selfTest = false;
};
struct State { double x, h, vx, v; };
State operator+(State a, State b) {
    return {a.x+b.x, a.h+b.h, a.vx+b.vx, a.v+b.v};
}
State operator*(double k, State a) { return {k*a.x,k*a.h,k*a.vx,k*a.v}; }

// Low-altitude, dry standard atmosphere, launch at sea level by default.
// Set launch elevation here for a real field; h is always height above launch.
double density(double h) {
    constexpr double launchElevationM = 0.0;
    const double altitude = launchElevationM + h;
    if (altitude < -500.0 || altitude > 11000.0)
        throw std::runtime_error("Atmosphere is only modeled from -500 to 11000 m ASL");
    const double temperature = 288.15 - 0.0065 * altitude;
    const double pressure = 101325.0 * std::pow(temperature / 288.15,
                                              cfg::G / (287.05 * 0.0065));
    return pressure / (287.05 * temperature);
}

double totalCd(double level) {
    level = std::clamp(level, 0.0, 9.0);
    const int lo = static_cast<int>(level);
    const int hi = std::min(lo + 1, 9);
    return cfg::BASE_CD + cfg::BRAKE_CD[lo] +
        (level - lo) * (cfg::BRAKE_CD[hi] - cfg::BRAKE_CD[lo]);
}

// Only valid for unpowered upward coast: constant rho and Cd, vertical motion.
// At/after apogee return the current height; do not square a descending velocity.
double predictedApogee(double h, double v, double cd,
                       double rho = cfg::PREDICTOR_RHO) {
    if (v <= 0.0) return h;
    const double b = 0.5 * rho * cd * cfg::AREA_M2;
    if (b < 1e-14) return h + v*v / (2.0 * cfg::G);
    return h + cfg::MASS_KG / (2.0*b) *
        std::log1p(b*v*v / (cfg::MASS_KG*cfg::G));
}
int commandIndex(double expected, double gain) {
    // Clamp before rounding so even a huge error cannot overflow an integer.
    return static_cast<int>(std::lround(std::clamp(
        gain * (expected - cfg::TARGET_M), 0.0, 9.0)));
}

// Random targets updated on a fixed physical clock, independent of dt.
// Smooth transitions avoid unphysical independent velocity kicks each step.
struct Weather {
    std::mt19937 rng;
    bool enabled;
    double meanWind, gustScale;
    std::array<double,3> from{0,0,0}, to{0,0,0};
    int interval = -1;
    explicit Weather(std::uint32_t seed, bool on, Conditions c = {})
        : rng(seed), enabled(on), meanWind(c.meanWind), gustScale(c.gustScale) {}
    double uniform(double a, double b) {
        return a + (b-a) * static_cast<double>(rng()) /
                          (static_cast<double>(std::mt19937::max()) + 1.0);
    }
    std::array<double,3> at(double t) {
        if (!enabled) return {0,0,0};
        const int desired = static_cast<int>(std::floor(t / cfg::GUST_PERIOD_S));
        while (interval < desired) {
            const double decay = std::exp(-cfg::GUST_PERIOD_S /
                                          cfg::GUST_TIME_CONSTANT_S);
            for (int i=0; i<3; ++i) from[i] = to[i] + (from[i]-to[i])*decay;
            to = {uniform(-1.5,1.5), uniform(-0.7,0.7), uniform(-0.15,0.15)};
            ++interval;
        }
        const double elapsed = t - interval*cfg::GUST_PERIOD_S;
        const double decay = std::exp(-elapsed/cfg::GUST_TIME_CONSTANT_S);
        std::array<double,3> out{};
        for (int i=0; i<3; ++i) out[i] = to[i] + (from[i]-to[i])*decay;
        for (double& component : out) component *= gustScale;
        out[0] += meanWind; // mean horizontal wind, m/s
        return out;    // horizontal wind, vertical wind, extra vertical acceleration
    }
};

State derivative(State s, double cd, std::array<double,3> weather,
                 bool variableDensity = true, Conditions c = {}) {
    const double rho = (variableDensity ? density(s.h) : cfg::PREDICTOR_RHO)*c.densityScale;
    const double ux = s.vx-weather[0], uz = s.v-weather[1];
    const double speed = std::hypot(ux,uz);
    const double factor = 0.5*rho*cd*c.dragScale*cfg::AREA_M2*speed/cfg::MASS_KG;
    return {s.vx, s.v, -factor*ux, -cfg::G-factor*uz+weather[2]};
}

// Handwritten fourth-order Runge-Kutta. Weather and actuator position are
// evaluated at RK stages. No numerical-integration library is used.
State rk4(State s, double t, double dt, double startLevel, int command,
          Weather weather, bool variableDensity = true, double fixedCd = -1.0,
          Conditions c = {}) {
    // Weather is copied: stage evaluations cannot advance the main RNG stream.
    auto f = [&](State q, double offset) {
        const double change = std::clamp(static_cast<double>(command)-startLevel,
            -cfg::SERVO_LEVELS_PER_S*offset, cfg::SERVO_LEVELS_PER_S*offset);
        const double cd = fixedCd >= 0 ? fixedCd : totalCd(startLevel+change);
        return derivative(q,cd,weather.at(t+offset),variableDensity,c);
    };
    const State k1 = f(s,0);
    const State k2 = f(s+(dt/2)*k1,dt/2);
    const State k3 = f(s+(dt/2)*k2,dt/2);
    const State k4 = f(s+dt*k3,dt);
    return s+(dt/6)*(k1+2*k2+2*k3+k4);
}
struct Result { double apogee, time; };
Result simulate(const Options& o, int mode, std::uint32_t seed,
                std::ostream* trace = nullptr, Conditions c = {}) {
    // mode: 0 = P control; 1 = always retracted; 2 = command maximum at t=0.
    State s{0,cfg::INITIAL_HEIGHT_M,0,o.impulse/cfg::MASS_KG};
    Weather weather(seed,o.disturbances,c);
    double t=0, level=0, nextControl=0, prediction=0;
    std::uint64_t tick=0;
    int command=0;
    auto log = [&] {
        if (!trace) return;
        Weather sample = weather;
        const auto wind = sample.at(t);
        const auto acceleration = derivative(s,totalCd(level),wind,true,c);
        *trace << t << '\t' << s.h << '\t' << s.v << '\t' << s.x << '\t'
            << s.vx << '\t' << acceleration.v << '\t' << acceleration.vx << '\t'
            << density(s.h)*c.densityScale << '\t' << prediction << '\t'
            << prediction-cfg::TARGET_M << '\t' << command << '\t' << level << '\t'
            << totalCd(level)*c.dragScale << '\t' << wind[0] << '\t' << wind[1] << '\t'
            << wind[2] << '\n';
    };
    if (trace) *trace << std::fixed << std::setprecision(6)
        << "time_s\theight_m\tvz_m_s\tx_m\tvx_m_s\taz_m_s2\tax_m_s2\t"
           "rho_kg_m3\tpred_apogee_m\tpred_error_m\tcommand\tactual_level\tCd\t"
           "wind_x_m_s\twind_z_m_s\textra_az_m_s2\n";
    while (t < cfg::MAX_TIME_S) {
        if (t >= nextControl-1e-12) {
            prediction=predictedApogee(s.h,s.v,o.predictBase ? cfg::BASE_CD : totalCd(level));
            command = mode==1 ? 0 : mode==2 ? 9 : commandIndex(prediction,o.gain);
            nextControl = static_cast<double>(++tick)*cfg::CONTROL_PERIOD_S;
        }
        log();
        // Land precisely on control updates, even when dt is not a divisor.
        const double dt=std::min({o.dt,nextControl-t,cfg::MAX_TIME_S-t});
        if (dt <= 0) throw std::runtime_error("Time integration stalled");
        State next=rk4(s,t,dt,level,command,weather,true,-1.0,c);
        double elapsed=dt;
        const bool apex = next.v <= 0.0;
        if (apex) {
            // Locate v=0 within the last step; never report a below-apex sample.
            double lo=0, hi=dt;
            for (int i=0;i<45;++i) {
                const double mid=(lo+hi)/2;
                if (rk4(s,t,mid,level,command,weather,true,-1.0,c).v>0) lo=mid; else hi=mid;
            }
            elapsed=(lo+hi)/2;
            next=rk4(s,t,elapsed,level,command,weather,true,-1.0,c);
        }
        level += std::clamp(static_cast<double>(command)-level,
            -cfg::SERVO_LEVELS_PER_S*elapsed,cfg::SERVO_LEVELS_PER_S*elapsed);
        t+=elapsed; s=next;
        weather.at(t); // commit RNG advancement once per accepted step
        if (apex) {
            s.v=0;
            prediction=predictedApogee(s.h,s.v,o.predictBase?cfg::BASE_CD:totalCd(level));
            log();
            return {s.h,t};
        }
    }
    throw std::runtime_error("Apogee not reached within time limit");
}

void selfTest() {
    auto require=[](bool ok,const char* msg) {
        if (!ok) throw std::runtime_error(std::string("TEST FAILED: ")+msg);
    };
    // Independent analytic reference at a fixed time (before apogee).
    for (double cd : {0.0,0.5,2.5}) {
        State s{0,12,0,90}; Weather calm(1,false);
        constexpr double dt=0.001, time=3.0;
        for (int i=0;i<3000;++i) s=rk4(s,i*dt,dt,0,0,calm,false,cd);
        double exactH,exactV;
        const double q=0.5*cfg::PREDICTOR_RHO*cd*cfg::AREA_M2/cfg::MASS_KG;
        if (q==0) { exactH=12+90*time-0.5*cfg::G*time*time; exactV=90-cfg::G*time; }
        else {
            const double theta=std::atan(90*std::sqrt(q/cfg::G));
            const double angle=theta-std::sqrt(cfg::G*q)*time;
            exactV=std::sqrt(cfg::G/q)*std::tan(angle);
            exactH=12+std::log(std::cos(angle)/std::cos(theta))/q;
        }
        require(std::abs(s.h-exactH)<1e-7 && std::abs(s.v-exactV)<1e-7,
                "RK4 versus exact vacuum/constant-density solution");
        const double invariant=predictedApogee(s.h,s.v,cd);
        require(std::abs(invariant-predictedApogee(12,90,cd))<1e-7,
                "analytical apogee invariant");
    }
    require(commandIndex(100,0.2)==0 && commandIndex(10000,0.2)==9,
            "controller saturation");
    require(commandIndex(cfg::TARGET_M+7.6,0.2)==2,"nearest-index rounding");
    require(predictedApogee(12,-4,0.5)==12,"descending prediction guard");
    require(density(500)<density(0),"density decreases with height");
    require(totalCd(0)==cfg::BASE_CD && totalCd(9)>totalCd(0),"drag table");
    Options a; a.disturbances=false;
    const auto coarse=simulate(a,0,42);
    a.dt=0.0005;
    const auto fine=simulate(a,0,42);
    require(std::abs(coarse.apogee-fine.apogee)<0.02,"controlled dt convergence");
    a.disturbances=true;
    const auto first=simulate(a,0,42), repeat=simulate(a,0,42);
    require(first.apogee==repeat.apogee,"seed repeatability");
    a.dt=0.001;
    require(std::abs(simulate(a,0,42).apogee-first.apogee)<0.02,
            "gust simulation dt convergence");
    a.disturbances=false;
    require(simulate(a,1,42).apogee>simulate(a,2,42).apogee,"braking authority");
    std::cout << "All self-tests passed (analytic references, bounds, density, "
                 "repeatability, braking, and timestep convergence).\n";
}

// Batch mode helpers are defined below; all results go to std::cout.
double promptNumber(const std::string&, double, double, double, bool = false);
void searchGain(Options);
void testConstantGain(Options);
void batchSelfTest();
void detailedFlights(const Options& o) {
        std::cout << std::fixed << std::setprecision(6)
                  << "Target: " << cfg::TARGET_M << " m; K: " << o.gain
                  << "; ideal initial velocity: " << o.impulse/cfg::MASS_KG << " m/s\n"
                  << "Mass: " << cfg::MASS_KG << " kg; reference area: " << cfg::AREA_M2
                  << " m^2; base Cd: " << cfg::BASE_CD << '\n'
                  << "Impulse: " << o.impulse << " N s; integration dt: " << o.dt
                  << " s; control period: " << cfg::CONTROL_PERIOD_S
                  << " s; servo speed: " << cfg::SERVO_LEVELS_PER_S << " levels/s\n"
                  << "Predictor density: " << cfg::PREDICTOR_RHO
                  << " kg/m^3; predictor Cd: "
                  << (o.predictBase?"retracted":"current actual setting") << '\n'
                  << "Disturbances: " << (o.disturbances?"on":"off")
                  << "; gust interval: " << cfg::GUST_PERIOD_S
                  << " s; gust time constant: " << cfg::GUST_TIME_CONSTANT_S << " s\n"
                  << "Extra brake Cd for settings 0..9:";
        for (double cd : cfg::BRAKE_CD) std::cout << ' ' << cd;
        std::cout << "\nEach table prints every integration step, plus the exact apex.\n"
                  << "Predicted apogee/error are held between controller updates.\n"
                  << "Without airbrakes retains body drag and the same weather.\n";
        std::array<double,2> sum{},sumSq{},worst{};
        for(int i=0;i<o.runs;++i) {
            const auto seed=o.seed+static_cast<std::uint32_t>(i);
            std::cout << "\nRUN " << i+1 << " / " << o.runs << "; SEED " << seed
                      << "\nWITH AIRBRAKES (PROPORTIONAL CONTROL)\n";
            const auto controlled=simulate(o,0,seed,&std::cout);
            std::cout << "\nWITHOUT AIRBRAKES (FULLY RETRACTED)\n";
            const auto retracted=simulate(o,1,seed,&std::cout);
            const std::array<Result,2> results{controlled,retracted};
            std::cout << "\nRUN " << i+1 << " SUMMARY; TARGET " << cfg::TARGET_M << " m\n";
            for (int j=0;j<2;++j) {
                const double error=results[j].apogee-cfg::TARGET_M;
                sum[j]+=error; sumSq[j]+=error*error;
                worst[j]=std::max(worst[j],std::abs(error));
                std::cout << (j==0?"With airbrakes:    ":"Without airbrakes: ")
                          << "apogee " << results[j].apogee << " m; target error "
                          << error << " m; time to apogee " << results[j].time << " s\n";
            }
            std::cout << "Apogee reduction from airbrakes: "
                      << retracted.apogee-controlled.apogee << " m\n";
            if(retracted.apogee<cfg::TARGET_M)
                std::cout << "Target above the retracted reference flight: brakes cannot add energy.\n";
        }
        if(o.runs>1) {
            std::cout << "\nAGGREGATE RESULTS FOR " << o.runs << " MATCHED FLIGHT PAIRS\n";
            for (int j=0;j<2;++j)
                std::cout << (j==0?"With airbrakes:    ":"Without airbrakes: ")
                    << "mean error " << sum[j]/o.runs << " m; RMSE "
                    << std::sqrt(sumSq[j]/o.runs) << " m; worst absolute error "
                    << worst[j] << " m\n";
        }
        std::cout.flush();
        if(!std::cout) throw std::runtime_error("Failed while writing terminal output");

}

int main(int argc,char** argv) {
    try {
        Options o;
        for (int i=1;i<argc;++i) {
            std::string arg=argv[i];
            auto value=[&]() { if (++i>=argc) throw std::runtime_error("Missing value for "+arg);
                              return std::string(argv[i]); };
            auto real=[&]() { const auto v=value(); std::size_t n; double x=std::stod(v,&n);
                if(n!=v.size() || !std::isfinite(x)) throw std::runtime_error("Invalid number"); return x; };
            if(arg=="--dt") o.dt=real();
            else if(arg=="--gain") o.gain=real();
            else if(arg=="--impulse") o.impulse=real();
            else if(arg=="--seed" || arg=="--runs") {
                const double n=real();
                if(n<0 || n>4294967295.0 || n!=std::floor(n)) throw std::runtime_error("Invalid integer");
                if(arg=="--seed") o.seed=static_cast<std::uint32_t>(n);
                else { if(n<1 || n>10000) throw std::runtime_error("Runs must be 1..10000"); o.runs=static_cast<int>(n); }
            }
            else if(arg=="--no-disturbances") o.disturbances=false;
            else if(arg=="--predict-base") o.predictBase=true;
            else if(arg=="--self-test") o.selfTest=true;
            else if(arg=="--help") {
                std::cout << "airbrake_sim [--gain K] [--impulse N_s] [--dt seconds] [--seed N]\n"
                    " [--runs N] [--no-disturbances] [--predict-base] [--self-test]\n"
                    "Default predictor uses current actual Cd; --predict-base uses retracted Cd.\n"
                    "A terminal menu selects K search, constant-K comparison, or full trajectories.\n"
                    "--gain applies to full trajectories; constant-K mode always uses cfg::K.\n"
                    "--runs sets the default batch size (otherwise 50) or detailed repeat count.\n"
                    "Batch modes vary launch impulse, density, drag, wind, and gust strength.\n"
                    "Without airbrakes means retracted brakes; body drag still applies.\n"
                    "Runs uses consecutive seeds with matching weather for each pair.\n";
                return 0;
            } else throw std::runtime_error("Unknown argument: "+arg);
        }
        if(o.dt<0.00001 || o.dt>0.01 || o.gain<0 || o.gain>100 || o.impulse<=0 || o.impulse>200)
            throw std::runtime_error("Require dt 0.00001..0.01 s, gain 0..100, impulse (0,200] N s");
        if(o.selfTest) { selfTest(); batchSelfTest(); return 0; }
        std::cout << "\nROCKET AIRBRAKE SIMULATOR\n"
                  << "1 - Find the best K across varied flight conditions\n"
                  << "2 - Compare fixed cfg::K against no airbrakes\n"
                  << "3 - Print full trajectories with and without airbrakes\n"
                  << "0 - Exit\n";
        const int choice=static_cast<int>(promptNumber("Selection",2,0,3,true));
        if (choice==0) return 0;
        if (choice==1) searchGain(o);
        else if (choice==2) testConstantGain(o);
        else detailedFlights(o);
    } catch(const std::exception& e) { std::cerr << "Error: " << e.what() << '\n'; return 1; }
}

// Read a whole input line so invalid input does not leave std::cin stuck.
// Press Enter for the default. EOF exits cleanly through the main error handler.
double promptNumber(const std::string& label, double fallback,
                    double minimum, double maximum, bool integer) {
    while (true) {
        std::cout << label << " [default " << fallback << "]: " << std::flush;
        std::string line;
        if (!std::getline(std::cin,line))
            throw std::runtime_error("Input ended before menu selection was complete");
        if (line.find_first_not_of(" \t\r")==std::string::npos) return fallback;
        std::istringstream in(line);
        double value;
        if ((in>>value) && (in>>std::ws).eof() && std::isfinite(value) &&
            value>=minimum && value<=maximum && (!integer || value==std::floor(value)))
            return value;
        std::cout << "Enter " << (integer?"a whole number":"a number") << " from "
                  << minimum << " to " << maximum << ".\n";
    }
}

struct Scenario {
    Options options;
    Conditions conditions;
    std::uint32_t seed;
};
// Independent random parameters for each flight; fixed for the duration of it.
// The same scenario objects are reused across all gains and both brake modes.
std::vector<Scenario> makeScenarios(Options base, int count, bool validation) {
    const std::uint32_t domain=validation?0x80000000u:0u;
    std::mt19937 generator((base.seed+domain)^0x9e3779b9u);
    auto uniform=[&](double lo,double hi) {
        return lo+(hi-lo)*static_cast<double>(generator()) /
            (static_cast<double>(std::mt19937::max())+1.0);
    };
    std::vector<Scenario> scenarios;
    for (int i=0;i<count;++i) {
        Scenario s{base,{},base.seed+domain+static_cast<std::uint32_t>(i)};
        s.options.impulse=base.impulse*uniform(0.92,1.08);
        s.conditions.densityScale=uniform(0.92,1.08);
        s.conditions.dragScale=uniform(0.90,1.10);
        s.conditions.meanWind=uniform(-5.0,5.0);
        s.conditions.gustScale=uniform(0.5,2.0);
        scenarios.push_back(s);
    }
    return scenarios;
}

struct Statistics {
    int n=0, hits=0;
    double sum=0, absSum=0, squareSum=0, worst=0;
    void add(double apogee,double tolerance) {
        const double error=apogee-cfg::TARGET_M;
        ++n; sum+=error; absSum+=std::abs(error); squareSum+=error*error;
        worst=std::max(worst,std::abs(error));
        if (std::abs(error)<=tolerance) ++hits;
    }
    double rmse() const { return n?std::sqrt(squareSum/n):0; }
    double mae() const { return n?absSum/n:0; }
    void print(const std::string& label,double tolerance) const {
        std::cout << label << ": flights " << n << "; mean error " << (n?sum/n:0)
            << " m; MAE " << mae() << " m; RMSE " << rmse()
            << " m; worst absolute error " << worst << " m; within +/-"
            << tolerance << " m: " << hits << '/' << n << " ("
            << (n?100.0*hits/n:0) << "%)\n";
    }
};

void describeBatch(const Options& o,int count,double tolerance) {
    std::cout << std::fixed << std::setprecision(4)
        << "\nTarget: " << cfg::TARGET_M << " m; flights per sample: " << count
        << "; tolerance: +/-" << tolerance << " m; base seed: " << o.seed
        << "; dt: " << o.dt << " s\n"
        << "Predictor: " << (o.predictBase?"retracted Cd":"current-position Cd")
        << ", fixed nominal density " << cfg::PREDICTOR_RHO << " kg/m^3\n"
        << "Uniform per-flight variations: impulse 92..108% of " << o.impulse
        << " N s; density 92..108%; actual Cd 90..110%;\n"
        << "mean horizontal wind -5..5 m/s; gust strength 50..200%.\n"
        << "Mass, geometry, servo speed, and predictor constants remain fixed.\n"
        << "Wind/gust disturbances: " << (o.disturbances?"enabled":"disabled")
        << "; impulse/density/Cd variation still applies.\n"
        << "Each comparison uses exactly matching conditions with and without airbrakes.\n"
        << "Batch output has one row per flight pair; mode 3 prints full trajectories.\n"
        << "All variations are illustrative assumptions, not measured distributions.\n";
}

// Print both flight results for each condition set, then aggregate paired scores.
void compareBatch(const std::vector<Scenario>& scenarios,double gain,double tolerance,
                  const std::string& label) {
    std::cout << '\n' << label << "; K = " << gain << '\n'
        << "flight\tseed\timpulse_Ns\trho_scale\tCd_scale\tmean_wind_m_s\tgust_scale\t"
           "with_apogee_m\twith_error_m\twith_time_s\twithout_apogee_m\twithout_error_m\t"
           "without_time_s\tcloser_to_target\n";
    Statistics controlledStats,retractedStats;
    int better=0,worse=0,tied=0,belowTarget=0;
    for (std::size_t i=0;i<scenarios.size();++i) {
        const auto& scenario=scenarios[i];
        Options o=scenario.options; o.gain=gain;
        const auto with=simulate(o,0,scenario.seed,nullptr,scenario.conditions);
        const auto without=simulate(o,1,scenario.seed,nullptr,scenario.conditions);
        controlledStats.add(with.apogee,tolerance);
        retractedStats.add(without.apogee,tolerance);
        const double withError=with.apogee-cfg::TARGET_M;
        const double withoutError=without.apogee-cfg::TARGET_M;
        const double difference=std::abs(withError)-std::abs(withoutError);
        const char* winner="tie";
        if (difference < -1e-9) { ++better; winner="with"; }
        else if (difference > 1e-9) { ++worse; winner="without"; }
        else ++tied;
        if (without.apogee<cfg::TARGET_M) ++belowTarget;
        const auto& c=scenario.conditions;
        std::cout << i+1 << '\t' << scenario.seed << '\t' << o.impulse << '\t'
            << c.densityScale << '\t' << c.dragScale << '\t' << c.meanWind << '\t'
            << c.gustScale << '\t' << with.apogee << '\t' << withError << '\t'
            << with.time << '\t' << without.apogee << '\t' << withoutError << '\t'
            << without.time << '\t' << winner << '\n';
    }
    controlledStats.print("WITH AIRBRAKES",tolerance);
    retractedStats.print("WITHOUT AIRBRAKES",tolerance);
    std::cout << "Airbrakes closer to target: " << better << "; worse: " << worse
              << "; tied: " << tied << '\n'
              << "Retracted flights already below target: " << belowTarget << '\n';
    if (retractedStats.rmse()>0)
        std::cout << "RMSE reduction: "
                  << 100*(1-controlledStats.rmse()/retractedStats.rmse()) << "%\n";
}

void testConstantGain(Options o) {
    // Deliberately ignore --gain: this mode tests the literal constant cfg::K.
    o.gain=cfg::K;
    std::cout << "\nFIXED-CONSTANT TEST: using cfg::K = " << cfg::K
              << " for every flight (no tuning).\n";
    const int count=static_cast<int>(promptNumber("Number of flight pairs",
                                                  std::min(o.runs>1?o.runs:50,1000),1,1000,true));
    const double tolerance=promptNumber("Target tolerance in metres",3,0,100);
    describeBatch(o,count,tolerance);
    compareBatch(makeScenarios(o,count,true),cfg::K,tolerance,"FIXED cfg::K RESULTS");
}

void searchGain(Options o) {
    std::cout << "\nK SEARCH: lowest training RMSE wins; smaller K wins exact ties.\n"
              << "The selected K is then checked on a separate, equally large sample.\n";
    const int count=static_cast<int>(promptNumber("Training flights per candidate K",
                                                  std::min(o.runs>1?o.runs:50,1000),1,1000,true));
    const double low=promptNumber("Smallest K to test",0,0,100);
    const double high=promptNumber("Largest K to test",std::max(2.0,low),low,100);
    const double step=promptNumber("K increment",0.1,0.0001,100);
    const double tolerance=promptNumber("Target tolerance in metres",3,0,100);
    const double span=(high-low)/step;
    if (span>499.0+1e-9)
        throw std::runtime_error("Search limited to 501 candidates; increase K increment");
    std::vector<double> gains;
    const int wholeSteps=static_cast<int>(std::floor(span+1e-10));
    for (int i=0;i<=wholeSteps;++i) gains.push_back(std::min(high,low+i*step));
    if (high-gains.back()>1e-10) gains.push_back(high);
    describeBatch(o,count,tolerance);
    const auto training=makeScenarios(o,count,false);
    Statistics baseline;
    for (const auto& s:training)
        baseline.add(simulate(s.options,1,s.seed,nullptr,s.conditions).apogee,tolerance);
    baseline.print("Training baseline WITHOUT AIRBRAKES",tolerance);
    std::cout << "\nTesting " << gains.size() << " candidate gains on identical training conditions.\n"
              << "K\tRMSE_m\tMAE_m\tmean_error_m\tworst_abs_error_m\thits\n";
    double bestK=low,bestScore=std::numeric_limits<double>::infinity();
    for (double gain:gains) {
        Statistics stats;
        for (const auto& s:training) {
            Options flight=s.options; flight.gain=gain;
            stats.add(simulate(flight,0,s.seed,nullptr,s.conditions).apogee,tolerance);
        }
        std::cout << gain << '\t' << stats.rmse() << '\t' << stats.mae() << '\t'
            << stats.sum/stats.n << '\t' << stats.worst << '\t' << stats.hits
            << '/' << stats.n << '\n' << std::flush;
        if (stats.rmse()<bestScore) { bestScore=stats.rmse(); bestK=gain; }
    }
    std::cout << "\nBest TESTED K: " << bestK << "; training RMSE: " << bestScore << " m\n";
    if (bestK==gains.front() || bestK==gains.back())
        std::cout << "Winner is on a search boundary; expand the range to investigate further.\n";
    std::cout << "This is a grid-search result for this model and condition distribution, "
                 "not a guaranteed global optimum.\n";
    // Validation conditions and weather seeds are disjoint from training.
    compareBatch(makeScenarios(o,count,true),bestK,tolerance,"UNSEEN VALIDATION RESULTS");
    std::cout << "\nTo adopt this gain, edit cfg::K in the source to "
              << std::setprecision(10) << bestK << " and recompile.\n"
              << "The search does not change cfg::K or write any files.\n";
}

void batchSelfTest() {
    auto check=[](bool ok,const char* message) {
        if (!ok) throw std::runtime_error(std::string("BATCH TEST FAILED: ")+message);
    };
    Options o;
    const auto training=makeScenarios(o,3,false), repeat=makeScenarios(o,3,false);
    const auto validation=makeScenarios(o,3,true);
    for (std::size_t i=0;i<training.size();++i) {
        const auto& s=training[i];
        check(s.options.impulse==repeat[i].options.impulse &&
              s.conditions.densityScale==repeat[i].conditions.densityScale,
              "scenario repeatability");
        check(s.options.impulse>=o.impulse*0.92 && s.options.impulse<=o.impulse*1.08 &&
              s.conditions.densityScale>=0.92 && s.conditions.densityScale<=1.08 &&
              s.conditions.dragScale>=0.90 && s.conditions.dragScale<=1.10 &&
              std::abs(s.conditions.meanWind)<=5 && s.conditions.gustScale>=0.5 &&
              s.conditions.gustScale<=2,"scenario bounds");
        check(s.seed!=validation[i].seed,"training/validation separation");
        Options zero=s.options; zero.gain=0;
        check(simulate(zero,0,s.seed,nullptr,s.conditions).apogee==
              simulate(zero,1,s.seed,nullptr,s.conditions).apogee,
              "paired zero-gain equality under variable conditions");
    }
    State sample{0,100,0,70};
    const auto nominal=derivative(sample,0.5,{0,0,0});
    Conditions denser; denser.densityScale=1.08;
    Conditions draggier; draggier.dragScale=1.10;
    check(derivative(sample,0.5,{0,0,0},true,denser).v<nominal.v &&
          derivative(sample,0.5,{0,0,0},true,draggier).v<nominal.v,
          "increased physical density and Cd increase upward drag");
    Options coarse=training[0].options;
    const auto first=simulate(coarse,0,training[0].seed,nullptr,training[0].conditions);
    coarse.dt/=2;
    const auto finer=simulate(coarse,0,training[0].seed,nullptr,training[0].conditions);
    check(std::abs(first.apogee-finer.apogee)<0.02,"variable-condition timestep convergence");
    Statistics stats;
    stats.add(cfg::TARGET_M+3,3); stats.add(cfg::TARGET_M-4,3);
    check(stats.n==2 && stats.hits==1 && stats.sum==-1 && stats.worst==4 &&
          stats.mae()==3.5 && std::abs(stats.rmse()-std::sqrt(12.5))<1e-12,
          "aggregate error statistics");
    std::cout << "Batch tests passed (conditions, repeatability, pairing, validation split, statistics).\n";
}
