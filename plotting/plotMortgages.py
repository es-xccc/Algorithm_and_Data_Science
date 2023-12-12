# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 20:22:23 2022

@author: tiw
"""
import pylab


# From Figure 8.9
def findPayment(loan, r, m):
    """Assumes: loan and r are floats, m an int
       Returns the monthly payment for a mortgage of size
       loan at a monthly rate of r for m months"""
    return loan * ((r * (1 + r) ** m) / ((1 + r) ** m - 1))


# Figure 11.6
class Mortgage(object):
    """Abstract class for building different kinds of mortgages"""

    def __init__(self, loan, annRate, months):
        self.loan = loan
        self.rate = annRate / 12.0
        self.months = months
        self.paid = [0.0]
        self.outstanding = [loan]
        self.payment = findPayment(loan, self.rate, months)
        self.legend = None  # description of mortgage

    def makePayment(self):
        self.paid.append(self.payment)
        reduction = self.payment - self.outstanding[-1] * self.rate
        self.outstanding.append(self.outstanding[-1] - reduction)

    def getTotalPaid(self):
        return sum(self.paid)

    def __str__(self):
        return self.legend

    def plotPayments(self, style):
        pylab.plot(self.paid[1:], style, label=self.legend)

    def plotBalance(self, style):
        pylab.plot(self.outstanding, style, label=self.legend)

    def plotTotPd(self, style):
        totPd = [self.paid[0]]
        for i in range(1, len(self.paid)):
            totPd.append(totPd[-1] + self.paid[i])
        pylab.plot(totPd, style, label=self.legend)

    def plotNet(self, style):
        totPd = [self.paid[0]]
        for i in range(1, len(self.paid)):
            totPd.append(totPd[-1] + self.paid[i])
        equityAcquired = pylab.array([self.loan] * \
                                     len(self.outstanding))
        equityAcquired = equityAcquired - \
                         pylab.array(self.outstanding)
        net = pylab.array(totPd) - equityAcquired
        pylab.plot(net, style, label=self.legend)


# Figure 11.7
class Fixed(Mortgage):
    def __init__(self, loan, r, months):
        Mortgage.__init__(self, loan, r, months)
        self.legend = 'Fixed, ' + str(r * 100) + '%'


class FixedWithPts(Mortgage):
    def __init__(self, loan, r, months, pts):
        Mortgage.__init__(self, loan, r, months)
        self.pts = pts
        self.paid = [loan * (pts / 100.0)]
        self.legend = 'Fixed, ' + str(r * 100) + '%, ' \
                      + str(pts) + ' points'


class TwoRate(Mortgage):
    def __init__(self, loan, r, months, teaserRate, teaserMonths):
        Mortgage.__init__(self, loan, teaserRate, months)
        self.teaserMonths = teaserMonths
        self.teaserRate = teaserRate
        self.nextRate = r / 12.0
        self.legend = str(teaserRate * 100) \
                      + '% for ' + str(self.teaserMonths) \
                      + ' months, then ' + str(r * 100) + '%'

    def makePayment(self):
        if len(self.paid) == self.teaserMonths + 1:
            self.rate = self.nextRate
            self.payment = findPayment(self.outstanding[-1],
                                       self.rate,
                                       self.months - self.teaserMonths)
        Mortgage.makePayment(self)


# Figure 11.8
def compareMortgages(amt, years, fixedRate, pts, ptsRate,
                     varRate1, varRate2, varMonths):
    totMonths = years * 12
    fixed1 = Fixed(amt, fixedRate, totMonths)
    fixed2 = FixedWithPts(amt, ptsRate, totMonths, pts)
    twoRate = TwoRate(amt, varRate2, totMonths, varRate1, varMonths)
    morts = [fixed1, fixed2, twoRate]
    for m in range(totMonths):
        for mort in morts:
            mort.makePayment()
    plotMortgages(morts, amt)


# Figure 11.9
def plotMortgages(morts, amt):
    def labelPlot(figure, title, xLabel, yLabel):
        pylab.figure(figure)
        pylab.title(title)
        pylab.xlabel(xLabel)
        pylab.ylabel(yLabel)
        pylab.legend(loc='best')

    styles = ['k-', 'k-.', 'k:']
    # Give names to figure numbers
    payments, cost, balance, netCost = 0, 1, 2, 3
    for i in range(len(morts)):
        pylab.figure(payments)
        morts[i].plotPayments(styles[i])
        pylab.figure(cost)
        morts[i].plotTotPd(styles[i])
        pylab.figure(balance)
        morts[i].plotBalance(styles[i])
        pylab.figure(netCost)
        morts[i].plotNet(styles[i])
        pylab.figure(4)
        pylab.xlim(0, 120)
        pylab.ylim(0, 175000)
        morts[i].plotTotPd(styles[i])

    labelPlot(payments, 'Monthly Payments of $' + str(amt) +
              ' Mortgages', 'Months', 'Monthly Payments')
    pylab.savefig('Monthly Payments of $.png')
    labelPlot(cost, 'Cash Outlay of $' + str(amt) +
              ' Mortgages', 'Months', 'Total Payments')
    pylab.savefig('Cash Outlay of $.png')
    labelPlot(balance, 'Balance Remaining of $' + str(amt) +
              ' Mortgages', 'Months', 'Remaining Loan Balance of $')
    pylab.savefig('Balance Remaining of $.png')
    labelPlot(netCost, 'Net Cost of $' + str(amt) + ' Mortgages',
              'Months', 'Payments - Equity $')
    pylab.savefig('Net Cost of $.png')
    labelPlot(4, 'enlarged_Cash Outlay of $' + str(amt) +
              ' Mortgages', 'Months', 'Total Payments')
    pylab.savefig('enlarged_Cash Outlay of $.png')


compareMortgages(amt=200000, years=30, fixedRate=0.07, pts=3.25, ptsRate=0.05, varRate1=0.045, varRate2=0.095,
                 varMonths=48)
# pylab.show()
