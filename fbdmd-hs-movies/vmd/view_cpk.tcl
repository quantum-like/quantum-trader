# Interactive VMD session with the same CPK look (no rendering):   vmd -e vmd/view_cpk.tcl -args traj/hs_charges_20p20m.lammpstrj [xray]
set traj  [lindex $argv 0]
set style [expr {[llength $argv] > 1 ? [lindex $argv 1] : "cpk"}]
package require pbctools
set molid [mol new $traj type lammpstrj waitfor all]
foreach {t e} {1 Ar 2 Na 3 Cl} { set s [atomselect $molid "type $t"]; $s set element $e; $s set name $e; $s delete }
color change rgb 0 0.15 0.35 0.90; color change rgb 1 0.85 0.15 0.15; color change rgb 2 0.78 0.78 0.80; color change rgb 8 0.97 0.97 0.98; color change rgb 16 0.40 0.40 0.42
color Display Background white; display backgroundgradient off
material add CPKms; material change ambient CPKms 0.30; material change diffuse CPKms 0.75; material change specular CPKms 0.25; material change shininess CPKms 0.55
material add Ghost; material change opacity Ghost 0.15; material change specular Ghost 0.05
set all [atomselect $molid all]; $all set radius 2.0; $all delete
while {[molinfo $molid get numreps] > 0} { mol delrep 0 $molid }
foreach {sel colid mat} [list "element Ar" 2 [expr {$style eq "xray" ? "Ghost" : "CPKms"}] "element Na" 1 CPKms "element Cl" 0 CPKms] {
    mol representation VDW 1.0 32.0; mol color ColorID $colid; mol selection $sel; mol material $mat; mol addrep $molid
}
display projection Perspective; display depthcue off; display shadows on; display ambientocclusion on; display aoambient 0.80; display aodirect 0.35
axes location Off
pbc box -color black -width 1.5 -style lines -molid $molid
display resetview; rotate y by 45; rotate x by 22   ;# if this looks from below, flip the sign here and in render_cpk.tcl (tilt)
animate style Loop; animate speed 0.6; animate forward
